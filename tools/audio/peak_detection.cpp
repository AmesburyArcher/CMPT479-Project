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

// TODO Changed peak detection kernel to the code below, should be nearly complete,
// processing byte stream rather than bit streams. Use this as a foundation and changes are probably needed but is solid basis.
// There is still some pseudocode in spots that need to be replaced
// Gonna have to deal with absolute values because .wav file can have negative values, could probably also track global min from the samples and then
// compare the abs(max) and abs(min) to see which is bigger to simplify logic


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
        if (inputStreams->getNumElements() != bitsPerSample) {
            throw std::invalid_argument(
                "bitsPerSample: " + std::to_string(bitsPerSample) +
                " != numInputStreams: " + std::to_string(inputStreams->getNumElements()));
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
        Value * splatMax = b.simd_fill(8, initialMax);

        b.CreateBr(combineLoop);

        b.SetInsertPoint(combineLoop);
        PHINode * blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
        blockOffsetPhi->addIncoming(sz_ZERO, entry);
        PHINode * maxVectorPhi = b.CreatePHI(b.fwVectorType(8), 2);
        maxVectorPhi->addIncoming(splatMax, entry);

        Value * newMax = maxVectorPhi;
        for (unsigned i = 0; i < 8; i++) {
            Value * bytepack1 = b.loadInputStreamPack("inputStreams", sz_ZERO, b.getInt32(i), blockOffsetPhi);
            newMax = b.CreateUMax(bytepack1, newMax);
        }
        Value * nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
        blockOffsetPhi->addIncoming(nextBlk, combineLoop);

        maxVectorPhi->addIncoming(newMax, combineDone);
        Value * moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);
        b.CreateCondBr(moreToDo, combineLoop, combineDone);

        b.SetInsertPoint(combineDone);

        // now newMax needs a horizontal max reduction
        Value * max2 = b.simd_umax(8, b.mvmd_srli(8, newMax, 1), newMax);
        Value * max3 = b.simd_umax(8, b.mvmd_srli(8, newMax, 2), max2);
        Value * max4 = b.simd_umax(8, b.mvmd_srli(8, newMax, 4), max3);
        Value * max5 = b.simd_umax(8, b.mvmd_srli(8, newMax, 8), max4);

        // for extracting the highest bit
        Value * maxToStore = b.CreateExtractElement(max5, b.getInt32(15)); //--------changed this
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
    StreamSet * monoStream = P.CreateStreamSet(bitsPerSample, 1);
    
    // Parse audio buffer (single channel)
    std::vector<StreamSet *> channels = {monoStream};
    ParseAudioBuffer(P, fileDescriptor, 1, bitsPerSample, channels, false);

    // Convert serial to parallel
    StreamSet* BasisBits = P.CreateStreamSet(bitsPerSample, 1);
    S2P(P, bitsPerSample, monoStream, BasisBits);
    SHOW_BIXNUM(BasisBits);

    std::cout << "Before kernel call" << std::endl;
    // Detect peak amplitude directly into the output scalar
    P.CreateKernelCall<PeakDetectionKernel>(bitsPerSample, BasisBits, peakAmplitude, initialAmplitude);

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
    bool isWav = true;
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

    auto fn = generatePipeline(driver, bitsPerSample);
    int32_t initialAmplitude = 0;
    int32_t peakAmplitude = 0;

    peakAmplitude = fn(fd, initialAmplitude);
    
    // Calculate maximum possible amplitude based on bits per sample
    int32_t maxPossibleAmplitude = (1 << (bitsPerSample - 1)) - 1;

    std::cout << "Peak amplitude: " << peakAmplitude << "\n";
    std::cout << "Maximum possible amplitude: " << maxPossibleAmplitude << "\n";

    double normalizationFactor = 1.0;
    if (peakAmplitude > 0) {
        normalizationFactor = static_cast<double>(maxPossibleAmplitude) / peakAmplitude;
    }
    
    std::cout << "Normalization factor: " << normalizationFactor << "\n";
    
    close(fd);
    return 0;
}