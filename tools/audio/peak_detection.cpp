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

class PeakDetectionKernel final : public MultiBlockKernel {
public:
    PeakDetectionKernel(LLVMTypeSystemInterface & b,
                         const unsigned int bitsPerSample,
                         StreamSet * const inputStreams,
                         Scalar * peakAmplitude)
    : MultiBlockKernel(b, "PeakDetectionKernel_" + std::to_string(bitsPerSample),
                      {Binding{"inputStreams", inputStreams, FixedRate(1)}},
                      {},
                      {Binding{"peakAmplitude", peakAmplitude}},
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
        // Set up the basic blocks for our loop
        BasicBlock * entry = b.GetInsertBlock();
        BasicBlock * loop = b.CreateBasicBlock("loop");
        BasicBlock * exit = b.CreateBasicBlock("exit");
        std::cout << "Inside kernel" << std::endl;

        // Constants and helper values
        Constant * const ZERO = b.getSize(0);
        Value * numOfBlocks = numOfStrides;

        // Initialize the max amplitude value
        Value * maxAmplitude = b.getScalarField("peakAmplitude");
        // initializing maxAmplitude to 0
        b.CreateStore(b.getSize(0), maxAmplitude);

        // Branch to the loop
        b.CreateBr(loop);

        // Main processing loop
        b.SetInsertPoint(loop);
        PHINode * blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
        blockOffsetPhi->addIncoming(ZERO, entry);

        // Load input streams (bit-parallel representation)
        std::vector<Value *> bitStreams(bitsPerSample);
        for (unsigned i = 0; i < bitsPerSample; i++) {
            bitStreams[i] = b.loadInputStreamBlock("inputStreams", b.getSize(i), blockOffsetPhi);
        }

        // Get the sign bit (MSB)
        Value * signBit = bitStreams[bitsPerSample - 1];

        // For all bits except the sign bit, apply two's complement conversion for negative values
        std::vector<Value *> processedBits(bitsPerSample - 1);
        for (unsigned i = 0; i < bitsPerSample - 1; i++) {
            // XOR with sign bit to flip bits for negative values
            processedBits[i] = b.CreateXor(bitStreams[i], signBit);
        }

        // For 2's complement, add 1 to negative values
        Value * carryIn = signBit; // Only add 1 to negative values 

        // Construct absolute value by doing addition with carry
        std::vector<Value *> absBits(bitsPerSample - 1);
        for (unsigned i = 0; i < bitsPerSample - 1; i++) {
            // Add with carry
            Value * sum = b.CreateXor(processedBits[i], carryIn);
            // Calculate carry out
            Value * carryOut = b.CreateAnd(processedBits[i], carryIn);
            absBits[i] = sum;
            // carryIn = carryOut;--------------------------------------------------------------
            carryIn = b.CreateOr(carryOut, carryIn); // Propagate carry correctly-----------------
        }

        // Now find the maximum value in this block
        Type * blockTy = bitStreams[0]->getType();
        // Value * blockMax = UndefValue::get(blockTy); ---------------------------------
        Value * isGreater = Constant::getNullValue(blockTy);

        Value * blockMax = absBits[0];  // Start with the first absolute value---------------------

        // Initialize blockMax to 0
        // blockMax = Constant::getNullValue(blockTy);---------------

        // MSB comparison logic
        for (int i = bitsPerSample - 2; i >= 0; i--) {
            Value * currentBit = absBits[i];
            Value * maxBit = blockMax;

            // If current bit is 1 and max bit is 0, set isGreater flag
            Value * newGreater = b.CreateAnd(b.CreateNot(maxBit), currentBit);
            isGreater = b.CreateOr(isGreater, newGreater);

            // Update blockMax for this bit position if isGreater
            // blockMax = b.CreateSelect(isGreater, currentBit, blockMax) ------------------------
            blockMax = b.CreateSelect(b.CreateICmpUGT(currentBit, blockMax), currentBit, blockMax); //-------
        }

        // reducing the blockMax, shifting right by 16bits, and then combing them w bitwise OR
        // Value * reducedMax = b.CreateOr(blockMax, b.CreateLShr(blockMax, 16));------------------
        // Value * blockMaxInt = b.hsimd_packl(bitsPerSample, reducedMax, reducedMax);-----------------
        // TRY ME! take out the loop below tho and uncomment the line below-----------------------------TRY ME---------------
        // Value * blockMaxInt = b.hsimd_packl(bitsPerSample, blockMax, blockMax);

        for (int shift = bitsPerSample / 2; shift > 0; shift /=2) {
            blockMax = b.CreateOr(blockMax, b.CreateLShr(blockMax, shift));
        }
        Value * blockMaxInt = b.CreateZExtOrTrunc(blockMax, b.getInt32Ty());
        

        // Compare with current max and update if needed
        Value * currentMax = b.CreateLoad(b.getInt32Ty(), maxAmplitude);
        Value * newMax = b.CreateSelect(
            b.CreateICmpUGT(blockMaxInt, currentMax),
            blockMaxInt,
            currentMax
        );
        // b.CreateStore(newMax, maxAmplitude);--------------------
        Value * finalMax = b.CreateZExtOrTrunc(newMax, b.getInt32Ty());
        b.CreateStore(finalMax, maxAmplitude);

        // Loop control
        Value * nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
        blockOffsetPhi->addIncoming(nextBlk, loop);
        Value * moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);

        b.CreateCondBr(moreToDo, loop, exit);
        // for debugging purposes
        llvm::errs() << "Final Block Max: " << blockMax << "\n";
        llvm::errs() << "Current Max Amplitude: " << currentMax << "\n";
        llvm::errs() << "New Max Amplitude Stored: " << newMax << "\n";


        // Exit block
        b.SetInsertPoint(exit);
    }

private:
    const unsigned int bitsPerSample;
    const unsigned int numInputStreams;
};


typedef void (*PipelineFunctionType)(StreamSetPtr & ss_buf, int32_t fd, int32_t & peak);

PipelineFunctionType generatePipeline(CPUDriver & pxDriver, const unsigned int &bitsPerSample) {
    std::cout << "Generating pipeline..." << std::endl;
    auto P = CreatePipeline(pxDriver,
        Output<streamset_t>("OutputBytes", 1, bitsPerSample, ReturnedBuffer(1)),
        Input<int32_t>("inputFileDecriptor"),
        Output<int32_t>("peakAmplitude"));

    StreamSet * OutputBytes = P.getOutputStreamSet("OutputBytes");
    Scalar * const fileDescriptor = P.getInputScalar("inputFileDecriptor");
    Scalar * peakAmplitude = P.getOutputScalar("peakAmplitude");

    // Create stream for the mono channel
    StreamSet * monoStream = P.CreateStreamSet(1, bitsPerSample);
    
    // Parse audio buffer (single channel)
    std::vector<StreamSet *> channels = {monoStream};
    ParseAudioBuffer(P, fileDescriptor, 1, bitsPerSample, channels, false);

    // Convert serial to parallel
    StreamSet* BasisBits = P.CreateStreamSet(bitsPerSample);
    S2P(P, bitsPerSample, monoStream, BasisBits);
    SHOW_BIXNUM(BasisBits);

    std::cout << "Before kernel call" << std::endl;
    // Detect peak amplitude directly into the output scalar
    P.CreateKernelCall<PeakDetectionKernel>(bitsPerSample, BasisBits, peakAmplitude);
    
    // Pass through the mono stream to output
    P2S(P, monoStream, OutputBytes);
    SHOW_BYTES(OutputBytes);
    ;
    auto compiledFn = P.compile();
    return reinterpret_cast<PipelineFunctionType>(compiledFn);
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
    StreamSetPtr wavStream;
    int32_t peakAmplitude = 0;

    fn(wavStream, fd, peakAmplitude);
    
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