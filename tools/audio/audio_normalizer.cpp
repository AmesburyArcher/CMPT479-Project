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

static cl::OptionCategory DemoOptions("Demo Options", "Demo control options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(DemoOptions));
static cl::opt<std::string> outputFile("o", cl::desc("Specify a file to save the modified .wav file."), cl::cat(DemoOptions));

class PeakDetectionKernel final : public MultiBlockKernel {
public:
    PeakDetectionKernel(LLVMTypeSystemInterface & b,
                         const unsigned int bitsPerSample,
                         StreamSet * const inputStreams,
                         Scalar * const peakAmplitude)
    : MultiBlockKernel(b, "PeakDetectionKernel_" + std::to_string(bitsPerSample),
                      {Binding{"inputStreams", inputStreams, FixedRate(1)}},
                      {},
                      {Binding{"peakAmplitude", peakAmplitude, InternalScalar()}},
                      {}, {})
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

        // Constants and helper values
        Constant * const ZERO = b.getSize(0);
        Value * numOfBlocks = numOfStrides;

        // Initialize the max amplitude value
        Value * maxAmplitude = b.getScalarField("peakAmplitude");

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
            carryIn = carryOut;
        }

        // Now find the maximum value in this block
        Type * blockTy = bitStreams[0]->getType();
        Value * blockMax = UndefValue::get(blockTy);
        Value * isGreater = Constant::getNullValue(blockTy);

        // Initialize blockMax to 0
        blockMax = Constant::getNullValue(blockTy);

        // MSB comparison logic
        for (int i = bitsPerSample - 2; i >= 0; i--) {
            Value * currentBit = absBits[i];
            Value * maxBit = blockMax;

            // If current bit is 1 and max bit is 0, set isGreater flag
            Value * newGreater = b.CreateAnd(b.CreateNot(maxBit), currentBit);
            isGreater = b.CreateOr(isGreater, newGreater);

            // Update blockMax for this bit position if isGreater
            blockMax = b.CreateSelect(isGreater, currentBit, blockMax);
        }

        // Convert the block max to a scalar
        // TODO This should take 3 arguments in the packl so investigate how to do this
        Value * blockMaxInt = b.hsimd_packl(1, blockMax);
        blockMaxInt = b.CreateZExtOrTrunc(blockMaxInt, b.getInt32Ty());

        // Compare with current max and update if needed
        Value * currentMax = b.CreateLoad(b.getInt32Ty(), maxAmplitude);
        Value * newMax = b.CreateSelect(
            b.CreateICmpUGT(blockMaxInt, currentMax),
            blockMaxInt,
            currentMax
        );
        b.CreateStore(newMax, maxAmplitude);

        // Loop control
        Value * nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
        blockOffsetPhi->addIncoming(nextBlk, loop);
        Value * moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);

        b.CreateCondBr(moreToDo, loop, exit);

        // Exit block
        b.SetInsertPoint(exit);
    }

private:
    const unsigned int bitsPerSample;
    const unsigned int numInputStreams;
};


typedef void (*PipelineFunctionType)(StreamSetPtr & ss_buf, int32_t fd);

PipelineFunctionType generatePipeline(CPUDriver & pxDriver, const unsigned int &numChannels, const unsigned int &bitsPerSample) {
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

    // Process each channel
    for (unsigned i = 0; i < numChannels; ++i) {
        // Convert serial to parallel
        StreamSet* BasisBits = P.CreateStreamSet(bitsPerSample);
        S2P(P, bitsPerSample, ChannelSampleStreams[i], BasisBits);
        SHOW_BIXNUM(BasisBits);

        // Normalize the audio using normalization kernel
        StreamSet *NormalizedBasisBits = P.CreateStreamSet(bitsPerSample);
        P.CreateKernelCall<NormalizePabloKernel>(bitsPerSample, BasisBits, NormalizedBasisBits);
        SHOW_BIXNUM(NormalizedBasisBits);

        // Convert back to serial
        NormalizedSampleStreams[i] = P.CreateStreamSet(1, bitsPerSample);
        P2S(P, NormalizedBasisBits, NormalizedSampleStreams[i]);
        SHOW_BYTES(NormalizedSampleStreams[i]);
    }

    P.CreateKernelCall<MergeKernel>(bitsPerSample, NormalizedSampleStreams[0], NormalizedSampleStreams[1], OutputBytes);

    SHOW_BYTES(OutputBytes);
    return P.compile();
}

int main(int argc, char *argv[])
{
    codegen::ParseCommandLineOptions(argc, argv, {&DemoOptions, codegen::codegen_flags()});

    CPUDriver driver("demo");
    const int fd = open(inputFile.c_str(), O_RDONLY);
    unsigned int sampleRate = 0, numChannels = 2, bitsPerSample = 8, numSamples = 0;
    bool isWav = true;
    try
    {
        readWAVHeader(fd, numChannels, sampleRate, bitsPerSample, numSamples);
        std::cout << "numChannels: " << numChannels << ", sampleRate: " << sampleRate << ", bitsPerSample: " << bitsPerSample << ", numSamples: " << numSamples << "\n";
        lseek(fd, 44, SEEK_SET);
    }
    catch (const std::exception &e)
    {
        llvm::errs() << "Warning: cannot parse " << inputFile << " WAV header for processing. Processing file as text.\n";
        lseek(fd, 0, SEEK_SET);
        isWav = false;
    }

    auto fn = generatePipeline(driver, numChannels, bitsPerSample);
    StreamSetPtr wavStream;

    
    fn(wavStream, fd);


    if (outputFile.getNumOccurrences() != 0) {
        const int fd_out = open(outputFile.c_str(), O_WRONLY | O_CREAT, 0666);
        if (LLVM_UNLIKELY(fd_out == -1)) {
            llvm::errs() << "Error: cannot write to " << outputFile << ".\n";
        } else {
            if (isWav) {
                auto header = createWAVHeader(numChannels, sampleRate, bitsPerSample, numSamples);
                write(fd_out, header.c_str(), header.size());
            }
            // NOTE: Despite a sample can be 8, 16, 32, etc. we treat the stream as bytestream (8-bit) to make it consistent with existing kernels.
            write(fd_out, wavStream.data<8>(), wavStream.length() * numChannels * (bitsPerSample / 8));
            close(fd_out);
        }
    }
    close(fd);
    return 0;
}