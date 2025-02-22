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

typedef void (*PipelineFunctionType)(StreamSetPtr & ss_buf, int32_t fd);

// This was moved to audio.cpp

// class NormalizePabloKernel final : public PabloKernel {
//     public:
//         NormalizePabloKernel(LLVMTypeSystemInterface & b, const unsigned int bitsPerSample,
//                              StreamSet * const inputStreams, Scalar * const gainFactor,
//                              StreamSet * const outputStreams)
//
//             : PabloKernel(b, "NormalizePabloKernel",
//                           {Binding{"input", inputStreams}, Binding{"gain", gainFactor}}, {Binding{"output", outputStreams}})
//             , mBitsPerSample(bitsPerSample) {}
//
//     protected:
//         void generatePabloMethod() override {
//             PabloBuilder pb(getEntryScope());
//
//             Var *input = getInput(0);
//
//             Var *gain = getInput(1);
//             Var *normalized = pb.createMul(input, gain); // this might need more adjustments
//
//
//             unsigned int maxAmplitude = (1 << (mBitsPerSample - 1)) - 1; // not sure about this part using pcm
//             Var *clamped = pb.createMin(normalized, pb.createConstant(maxAmplitude));
//
//             Var *output = getOutput(0);
//             pb.createAssign(output, clamped);
//         }
//
//     private:
//         unsigned int mBitsPerSample;
//     };

// PipelineFunctionType generatePipeline(CPUDriver & pxDriver, const unsigned int &numChannels, const unsigned int &bitsPerSample)
// {
//
//     auto P = CreatePipeline(pxDriver, Output<streamset_t>("OutputBytes", 1, bitsPerSample * numChannels, ReturnedBuffer(1)), Input<int32_t>("inputFileDecriptor"));
//
//     StreamSet * OutputBytes = P.getOutputStreamSet("OutputBytes");
//
//     Scalar * const fileDescriptor = P.getInputScalar("inputFileDecriptor");
//
//     std::vector<StreamSet *> ChannelSampleStreams(numChannels);
//     for (unsigned i=0;i<numChannels;++i)
//     {
//         ChannelSampleStreams[i] = P.CreateStreamSet(1,bitsPerSample);
//     }
//
//     ParseAudioBuffer(P, fileDescriptor, numChannels, bitsPerSample, ChannelSampleStreams);
//
//     std::vector<StreamSet *> NormalizedSampleStreams(numChannels);
//
//     for (unsigned i = 0; i < numChannels; ++i)
//     {
//         StreamSet* BasisBits = P.CreateStreamSet(bitsPerSample);
//         S2P(P, bitsPerSample, ChannelSampleStreams[i], BasisBits);
//         SHOW_BIXNUM(BasisBits);
//
//         // Create scalars don't appear to work the way you're calling them here
//
//         // Scalar *peakAmplitude = P.CreateScalar("PeakAmplitude", bitsPerSample); // need peak detection for proper normalizing
//         // P.CreateKernelCall<PeakDetectionKernel>(bitsPerSample, BasisBits, peakAmplitude);
//
//         // Scalar *gainFactor = P.CreateScalar("GainFactor", bitsPerSample);
//         // Scalar *gainFactor = P.CreateScalar(P.getInt64Ty());
//         // P.CreateKernelCall<ComputeGainKernel>(peakAmplitude, gainFactor); // this needs further edge case testing
//
//         StreamSet *NormalizedBasisBits = P.CreateStreamSet(bitsPerSample);
//         P.CreateKernelCall<NormalizePabloKernel>(bitsPerSample, BasisBits, nullptr, NormalizedBasisBits);
//         //SHOW_STREAM(NormalizedBasisBits);
//
//         NormalizedSampleStreams[i] = P.CreateStreamSet(1, bitsPerSample);
//         P2S(P, NormalizedBasisBits, NormalizedSampleStreams[i]);
//         //SHOW_BYTES(OutputStreams[i]);
//     }
//     //so that it works with mono too --------------- still needs thorough testing
//     // Current merge kernel needs 5 arguments so have to change that before we handle only 1 channel
//     // if (numChannels== 1) {
//     //     P.CreateKernelCall<MergeKernel>(bitsPerSample, NormalizedSampleStreams[0], OutputBytes);
//     // } else {
//     //     P.CreateKernelCall<MergeKernel>(bitsPerSample, NormalizedSampleStreams[0], NormalizedSampleStreams[1], OutputBytes);
//     // }
//     P.CreateKernelCall<MergeKernel>(bitsPerSample, NormalizedSampleStreams[0], NormalizedSampleStreams[1], OutputBytes);
//
//     SHOW_BYTES(OutputBytes);
//     return P.compile();
// }

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