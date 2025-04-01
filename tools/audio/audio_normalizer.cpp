#include <cstdio>
#include <vector>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Module.h>
#include <re/adt/re_name.h>
#include <re/adt/re_re.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/core/streamsetptr.h>
#include <kernel/scan/scanmatchgen.h>
#include <kernel/streamutils/stream_select.h>
#include <string>
#include <toolchain/toolchain.h>
#include <fcntl.h>
#include <iostream> 
#include <kernel/pipeline/driver/cpudriver.h>
#include <audio/audio.h>
#include <audio/stream_manipulation.h>
#include <iostream>
#include <boost/intrusive/detail/math.hpp>
#include <util/aligned_allocator.h>
#include <kernel/pipeline/program_builder.h>
#include <cmath>



using namespace kernel;
using namespace llvm;
using namespace codegen;
using namespace audio;

#define SHOW_STREAM(name)           \
    if (codegen::EnableIllustrator) \
    P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name)           \
    if (codegen::EnableIllustrator) \
    P.captureBixNum(#name, name)
#define SHOW_BYTES(name)            \
    if (codegen::EnableIllustrator) \
    P.captureByteData(#name, name)

static cl::OptionCategory DemoOptions("Demo Options", "Demo control options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(DemoOptions));
static cl::opt<std::string> outputFile("o", cl::desc("Specify a file to save the modified .wav file."), cl::cat(DemoOptions));

int countFractionalDigits(double value) {
    std::string str = std::to_string(value);
    size_t pos = str.find('.');
    if (pos == std::string::npos) return 0;

    str.erase(str.find_last_not_of('0') + 1, std::string::npos);

    return str.length() - pos - 1;
}

typedef void (*PipelineFunctionType)(StreamSetPtr & ss_buf, int32_t fd);

PipelineFunctionType generateNormalizationPipeline(CPUDriver & pxDriver, const unsigned int &numChannels, const unsigned int &bitsPerSample, double normalizationFactor) {
    auto P = CreatePipeline(pxDriver,
        Output<streamset_t>("OutputBytes", 1, bitsPerSample * numChannels, ReturnedBuffer(1)),
        Input<int32_t>("inputFileDecriptor"));

    StreamSet * OutputBytes = P.getOutputStreamSet("OutputBytes");
    Scalar * const fileDescriptor = P.getInputScalar("inputFileDecriptor");


    // Create streams for each channel
    std::vector<StreamSet *> ChannelSampleStreams(numChannels);
    for (unsigned i = 0; i < numChannels; ++i) {
        ChannelSampleStreams[i] = P.CreateStreamSet(1, bitsPerSample);
    }

    // Parse audio buffer into channels
    ParseAudioBuffer(P, fileDescriptor, numChannels, bitsPerSample, ChannelSampleStreams, true);

    std::vector<StreamSet *> NormalizedSampleStreams(numChannels);

    std::cout << "Normalization factor: " << normalizationFactor << std::endl;

    int precision = countFractionalDigits(normalizationFactor);

    std::cout << "Precision: " << precision << std::endl;

    // Process each channel
    for (unsigned i = 0; i < numChannels; ++i) {
        // Convert serial to parallel
        StreamSet* BasisBits = P.CreateStreamSet(bitsPerSample);
        S2P(P, bitsPerSample, ChannelSampleStreams[i], BasisBits);
        SHOW_BIXNUM(BasisBits);

        // Normalize the audio using normalization kernel
        StreamSet *NormalizedBasisBits = P.CreateStreamSet(bitsPerSample);
        P.CreateKernelCall<NormalizePabloKernel>(bitsPerSample, BasisBits, normalizationFactor, precision, NormalizedBasisBits);
        SHOW_BIXNUM(NormalizedBasisBits);

        // Convert back to serial
        NormalizedSampleStreams[i] = P.CreateStreamSet(1, bitsPerSample);
        if ( numChannels == 1){
            P2S(P, NormalizedBasisBits, OutputBytes);
        } else {
            P2S(P, NormalizedBasisBits, NormalizedSampleStreams[i]);
        }
        SHOW_BYTES(NormalizedSampleStreams[i]);
    }

    if (numChannels == 2) {
        P.CreateKernelCall<MergeKernel>(bitsPerSample, NormalizedSampleStreams[0], NormalizedSampleStreams[1], OutputBytes);
    }
    
    SHOW_BYTES(OutputBytes);
    return P.compile();
}

class PeakDetectionKernel final : public MultiBlockKernel {
public:
    PeakDetectionKernel(LLVMTypeSystemInterface & b,
                         const unsigned int bitsPerSample,
                         StreamSet * const inputStreams,
                         Scalar * peakAmplitude,
                         Scalar * initialAmplitude)
    : MultiBlockKernel(b, "PeakDetectionKernel_" + std::to_string(bitsPerSample),
                      {Binding{"inputStreams", inputStreams, FixedRate(1)}},
                      {},
                      {Binding{"initialAmplitude", initialAmplitude}},
                      {Binding{"peakAmplitude", peakAmplitude}},
                      {})
    , bitsPerSample(bitsPerSample)
    , numInputStreams(inputStreams->getNumElements())
    {
        if (inputStreams->getNumElements() != 1) {
            throw std::invalid_argument(
                "Input stream must be full byte stream");
        }
    }


protected:
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override {
        BasicBlock * entry = b.GetInsertBlock();
        BasicBlock * combineLoop = b.CreateBasicBlock("combineLoop");
        BasicBlock * combineDone = b.CreateBasicBlock("combineDone");
        Constant * const sz_ZERO = b.getSize(0);

        Value * numOfBlocks = numOfStrides;
        if (getStride() != b.getBitBlockWidth()) {
            numOfBlocks = b.CreateShl(numOfStrides, b.getSize(boost::intrusive::detail::floor_log2(getStride()/b.getBitBlockWidth())));
        }

        Value * initialMax = b.getScalarField("initialAmplitude");
        Value * splatMax = b.simd_fill(bitsPerSample, initialMax);

        b.CreateBr(combineLoop);

        b.SetInsertPoint(combineLoop);
        PHINode * blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
        blockOffsetPhi->addIncoming(sz_ZERO, entry);
        PHINode * maxVectorPhi = b.CreatePHI(b.fwVectorType(bitsPerSample), 2);
        maxVectorPhi->addIncoming(splatMax, entry);

        Value * newMax = maxVectorPhi;
        if (bitsPerSample == 8) {
            // For 8-bit samples, process all 8 byte packs
            for (unsigned i = 0; i < 8; i++) {
                Value * bytepack = b.loadInputStreamPack("inputStreams", sz_ZERO, b.getInt32(i), blockOffsetPhi);
                Value * samples = b.CreateBitCast(bytepack, b.fwVectorType(8));
                newMax = b.CreateUMax(samples, newMax);
            }
        } else {

            for (unsigned i = 0; i < 4; i++) {
                Value * bytepack1 = b.loadInputStreamPack("inputStreams", sz_ZERO, b.getInt32(i*2), blockOffsetPhi);
                Value * bytepack2 = b.loadInputStreamPack("inputStreams", sz_ZERO, b.getInt32(i*2+1), blockOffsetPhi);

                // Value * combined = b.CreateOr(b.CreateShl(bytepack2, 8), bytepack1);
                // Value * combined = b.CreateOr(b.CreateShl(bytepack2, b.getInt64(8)), bytepack1); // shifting by 8 btis now
                Value * shiftAmount = b.simd_fill(64, b.getInt64(8));


                Value * shifted = b.CreateShl(bytepack2, shiftAmount);
                Value * combined = b.CreateOr(shifted, bytepack1);

                Value * samples = b.CreateBitCast(combined, b.fwVectorType(16));

                // Get absolute value for signed samples
                Value * zeroVec = b.simd_fill(16, b.getInt16(0)); //creating 16 lanes of 0s
                Value * isNegative = b.CreateICmpSLT(samples, zeroVec); //returns 1 if true
                Value * absSamples = b.CreateSelect(isNegative, b.CreateNeg(samples), samples);

                newMax = b.CreateUMax(absSamples, newMax);
            }

        }
        Value * nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
        blockOffsetPhi->addIncoming(nextBlk, combineLoop);

        maxVectorPhi->addIncoming(newMax, combineLoop);
        Value * moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);
        b.CreateCondBr(moreToDo, combineLoop, combineDone);

        b.SetInsertPoint(combineDone);

        // Store the final value
        Value * currentMax = newMax;
        unsigned lanes = b.getBitBlockWidth() / bitsPerSample;
        unsigned logSteps = static_cast<unsigned>(std::log2(lanes));

        for (unsigned i = 0; i < logSteps; i++) {
            unsigned shiftAmount = 1 << i;
            Value * shifted = b.mvmd_srli(bitsPerSample, currentMax, shiftAmount);
            currentMax = b.simd_umax(bitsPerSample, shifted, currentMax);
        }

        // Extracting the max element
        Value* maxToStoreRaw = b.CreateExtractElement(currentMax, b.getInt32(0));

        Value * maxToStore = b.CreateZExt(maxToStoreRaw, b.getInt32Ty());

        b.setScalarField("peakAmplitude", maxToStore);
    }

private:
    const unsigned int bitsPerSample;
    const unsigned int numInputStreams;
};


typedef int32_t (*PipelineFn)(int32_t fd, int32_t initialAmplitude);

PipelineFn generatePeakPipeline(CPUDriver & pxDriver, const unsigned int &bitsPerSample) {
    std::cout << "Generating pipeline..." << std::endl;
    auto P = CreatePipeline(pxDriver,
        Input<int32_t>("inputFileDescriptor"),
        Input<int32_t>("initialAmplitude"),
        Output<int32_t>("peakAmplitude"));

    Scalar * const fileDescriptor = P.getInputScalar("inputFileDescriptor");
    Scalar * peakAmplitude = P.getOutputScalar("peakAmplitude");
    Scalar * initialAmplitude = P.getInputScalar("initialAmplitude");

    // Create stream for the mono channel
    StreamSet * monoStream = P.CreateStreamSet(1, bitsPerSample);

    // Parse audio buffer (single channel)
    std::vector<StreamSet *> channels = {monoStream};
    ParseAudioBuffer(P, fileDescriptor, 1, bitsPerSample, channels, false);


    std::cout << "Before kernel call" << std::endl;
    // Detect peak amplitude directly into the output scalar
    P.CreateKernelCall<PeakDetectionKernel>(bitsPerSample, monoStream, peakAmplitude, initialAmplitude);

    return P.compile();
}


int main(int argc, char *argv[])
{
    codegen::ParseCommandLineOptions(argc, argv, {&DemoOptions, codegen::codegen_flags()});

    CPUDriver driver("demo");
    const int fd = open(inputFile.c_str(), O_RDONLY);
    unsigned int sampleRate = 0, numChannels = 2, bitsPerSample = 8, numSamples = 0;
    try {
        readWAVHeader(fd, numChannels, sampleRate, bitsPerSample, numSamples);
        std::cout << "WAV File Info: " << numChannels << " channels, "
                 << sampleRate << " Hz, "
                 << bitsPerSample << " bits per sample, "
                 << numSamples << " samples\n";

        // For now, we only handle mono files
        // if (numChannels != 1) {
        //     llvm::errs() << "Error: This tool only works with mono (1-channel) WAV files.\n";
        //     close(fd);
        //     return 1;
        // }

        lseek(fd, 44, SEEK_SET);
    } catch (const std::exception &e) {
        llvm::errs() << "Error: " << inputFile << " is not a valid WAV file.\n";
        close(fd);
        return 1;
    }

    /*** === ADDITION: Basic C-based Peak Detection === ***/
    auto c_start = std::chrono::high_resolution_clock::now();
    FILE *wavFile = fopen(inputFile.c_str(), "rb");
    if (!wavFile) {
        std::cerr << "Error: Unable to open WAV file for peak detection.\n";
        close(fd);
        return 1;
    }

    fseek(wavFile, 44, SEEK_SET);  // Skip WAV header
    uint8_t sample_8t;
    int16_t sample_16t;
    uint32_t peakAmplitude_C = 0;

    if (bitsPerSample == 8) {
        while (fread(&sample_8t, sizeof(uint8_t), 1, wavFile) == 1) {
            if (sample_8t > peakAmplitude_C) {
                peakAmplitude_C = sample_8t;
            }
        }
    } else if (bitsPerSample == 16) {
        while (fread(&sample_16t, sizeof(int16_t), 1, wavFile) == 1) {
            if (abs(sample_16t) > peakAmplitude_C) {
                peakAmplitude_C = abs(sample_16t);
            }
        }
    }
    fclose(wavFile);

    auto c_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> c_duration = c_end - c_start;

    std::cout << "Peak Amplitude (Basic C Detection): " << (int)peakAmplitude_C << "\n";
    std::cout << "C implementation time: " << c_duration.count() << " ms\n\n";

    auto simd_start = std::chrono::high_resolution_clock::now();
    auto fn_peak = generatePeakPipeline(driver, bitsPerSample);
    int32_t initialAmplitude = 0;
    int32_t peakAmplitude = fn_peak(fd, initialAmplitude);

    auto simd_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> simd_duration = simd_end - simd_start;

    // Calculate maximum possible amplitude based on bits per sample
    int32_t maxPossibleAmplitude = bitsPerSample == 8 ? (1 << bitsPerSample) : (1 << (bitsPerSample - 1)) - 1;

    std::cout << "(SIMD) implementation time: " << simd_duration.count() << " ms\n";
    std::cout << "(SIMD) Peak amplitude: " << peakAmplitude << "\n";
    std::cout << "Maximum possible amplitude: " << maxPossibleAmplitude << "\n";

    double normalizationFactor = 1.0;
    if (peakAmplitude > 0) {
        normalizationFactor = static_cast<double>(maxPossibleAmplitude) / peakAmplitude;
    }

    // resetting the offset for the normalization process
    lseek(fd, 44, SEEK_SET);

    auto fn = generateNormalizationPipeline(driver, numChannels, bitsPerSample, normalizationFactor);
    StreamSetPtr wavStream;

    fn(wavStream, fd);

    if (outputFile.getNumOccurrences() != 0) {
        const int fd_out = open(outputFile.c_str(), O_WRONLY | O_CREAT, 0666);
        if (LLVM_UNLIKELY(fd_out == -1)) {
            llvm::errs() << "Error: cannot write to " << outputFile << ".\n";
        } else {
            auto header = createWAVHeader(numChannels, sampleRate, bitsPerSample, numSamples);
            write(fd_out, header.c_str(), header.size());

            // NOTE: Despite a sample can be 8, 16, 32, etc. we treat the stream as bytestream (8-bit) to make it consistent with existing kernels.
            write(fd_out, wavStream.data<8>(), wavStream.length() * numChannels * (bitsPerSample / 8));
            close(fd_out);
        }
    }
    close(fd);
    return 0;
}