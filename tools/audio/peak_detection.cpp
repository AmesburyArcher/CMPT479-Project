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
#include <chrono>

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

static cl::OptionCategory PeakDetectionOptions("Peak Detection Options", "Peak detection control options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(PeakDetectionOptions));

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
            // For 16-bit samples, process 4 pairs of byte packs
            for (unsigned i = 0; i < 4; i++) {
                Value * bytepack1 = b.loadInputStreamPack("inputStreams", sz_ZERO, b.getInt32(i*2), blockOffsetPhi);
                Value * bytepack2 = b.loadInputStreamPack("inputStreams", sz_ZERO, b.getInt32(i*2+1), blockOffsetPhi);

                Value * combined = b.CreateOr(b.CreateShl(bytepack2, 8), bytepack1);

                Value * samples = b.CreateBitCast(combined, b.fwVectorType(16));

                // Get absolute value for signed samples
                Value * absSamples = b.CreateSelect(
                    b.CreateICmpSLT(samples, b.getInt16(0)),
                    b.CreateNeg(samples),
                    samples
                );

                newMax = b.CreateUMax(absSamples, newMax);
            }
        }
        Value * nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
        blockOffsetPhi->addIncoming(nextBlk, combineLoop);

        maxVectorPhi->addIncoming(newMax, combineLoop);
        Value * moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);
        b.CreateCondBr(moreToDo, combineLoop, combineDone);

        b.SetInsertPoint(combineDone);

        // Horizontal max reduction for newMax
        // Value * max2 = bitsPerSample == 8 ? b.simd_umax(8, b.mvmd_srli(8, newMax, 1), newMax) : b.simd_umax(16, b.mvmd_srli(16, newMax, 1), newMax);
        // Value * max3 = bitsPerSample == 8 ? b.simd_umax(8, b.mvmd_srli(8, newMax, 2), max2) : b.simd_umax(16, b.mvmd_srli(16, newMax, 2), max2);
        // Value * max4 = bitsPerSample == 8 ? b.simd_umax(8, b.mvmd_srli(8, newMax, 4), max3) : b.simd_umax(16, b.mvmd_srli(16, newMax, 4), max3);
        // Value * max5 = bitsPerSample == 8 ? b.simd_umax(8, b.mvmd_srli(8, newMax, 8), max4) : max4;

        // Value * maxToStore = b.CreateExtractElement(max5, b.getInt32(bitsPerSample == 8 ? 15 : 7));

        // Store the final value
        Value * currentMax = newMax;
        unsigned lanes = b.getBitBlockWidth() / bitsPerSample;
        unsigned logSteps = static_cast<unsigned>(std::log2(lanes));

        for (unsigned i = 0; i < logSteps; i++) {
            unsigned shiftAmount = 1 << i;
            Value * shifted = b.mvmd_srli(bitsPerSample, currentMax, shiftAmount);
            currentMax = b.simd_umax(bitsPerSample, shifted, currentMax);
            // b.CallPrintRegister("max_step_" + std::to_string(i), currentMax);
        }


        // now newMax needs a horizontal max reduction
        // Value * max2 = b.simd_umax(8, b.mvmd_srli(8, newMax, 1), newMax);
        // Value * max3 = b.simd_umax(8, b.mvmd_srli(8, newMax, 2), max2);
        // Value * max4 = b.simd_umax(8, b.mvmd_srli(8, newMax, 4), max3);
        // Value * max5 = b.simd_umax(8, b.mvmd_srli(8, newMax, 8), max4);

        // for extracting the highest bit
        // Value * maxToStore = b.CreateExtractElement(max5, b.getInt32(15));
        Value * maxToStoreRaw = b.CreateExtractElement(currentMax, b.getInt32(0));

        Value * maxToStore = b.CreateZExt(maxToStoreRaw, b.getInt32Ty());


        b.setScalarField("peakAmplitude", maxToStore);
    }

private:
    const unsigned int bitsPerSample;
    const unsigned int numInputStreams;
};


typedef int32_t (*PipelineFn)(int32_t fd, int32_t initialAmplitude);

PipelineFn generatePipeline(CPUDriver & pxDriver, const unsigned int &bitsPerSample) {
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
    codegen::ParseCommandLineOptions(argc, argv, {&PeakDetectionOptions, codegen::codegen_flags()});

    CPUDriver driver("peak_detection");
    const int fd = open(inputFile.c_str(), O_RDONLY);
    if (fd == -1) {
        llvm::errs() << "Error: cannot open " << inputFile << ".\n";
        return 1;
    }

    unsigned int sampleRate = 0, numChannels = 1, bitsPerSample = 8, numSamples = 0;
    try {
        readWAVHeader(fd, numChannels, sampleRate, bitsPerSample, numSamples);
        std::cout << "WAV File Info: " << numChannels << " channels, "
                 << sampleRate << " Hz, " 
                 << bitsPerSample << " bits per sample, "
                 << numSamples << " samples\n";
        
        // For now, we only handle mono files
        if (numChannels != 1) {
            llvm::errs() << "Error: This tool only works with mono (1-channel) WAV files.\n";
            close(fd);
            return 1;
        }
        
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

    auto fn = generatePipeline(driver, bitsPerSample);
    int32_t initialAmplitude = 0;
    int32_t peakAmplitude = 0;

    peakAmplitude = fn(fd, initialAmplitude);

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
    
    std::cout << "Normalization factor: " << normalizationFactor << "\n";
    
    close(fd);
    return 0;
}