#pragma once
#include <vector>
#include <fstream>
#include <string>
#include <memory>
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/core/relationship.h>
#include <kernel/io/source_kernel.h>
#include <pablo/builder.hpp>
#include <util/aligned_allocator.h>

using namespace kernel;
using namespace llvm;
using namespace pablo;

namespace audio
{
    void readWAVHeader(
        const int &fd,
        unsigned int &numChannels,
        unsigned int &sampleRate,
        unsigned int &bitsPerSample,
        unsigned int &numSamples);

    void readTextFile(const int &fd, std::vector<int8_t, AlignedAllocator<int8_t, 64>>& buffer);

    std::string createWAVHeader(
        const unsigned int &numChannels,
        const unsigned int &sampleRate,
        const unsigned int &bitsPerSample,
        const unsigned int &numSamples);

    void ParseAudioBuffer(
        const std::unique_ptr<ProgramBuilder> &P,
        Scalar *const fileDescriptor,
        unsigned int numChannels,
        unsigned int bitsPerSample,
        std::vector<StreamSet *> &outputDataStreams,
        const bool& splitChannels = true);

    void ParseAudioBuffer(
        const std::unique_ptr<ProgramBuilder> &P,
        Scalar *const buffer,
        Scalar *const length,
        unsigned int numChannels,
        unsigned int bitsPerSample,
        std::vector<StreamSet *> &outputDataStreams,
        const bool& splitChannels = true);

    void S2P(
        const std::unique_ptr<ProgramBuilder> &P,
        unsigned int bitsPerSample,
        StreamSet * const inputStream,
        StreamSet *&outputStreams);

    void P2S(
        const std::unique_ptr<ProgramBuilder> &P,
        StreamSet * const inputStreams,
        StreamSet *&outputStream);

    class FlexS2PKernel final : public MultiBlockKernel {
    public:
        FlexS2PKernel(kernel::KernelBuilder & b, 
                const unsigned int bitsPerSample,
                StreamSet * const inputStream,
                StreamSet * const outputStreams);
    protected:
        void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
    private:
        unsigned int bitsPerSample;
    };


    class DiscontinuityKernel final : public PabloKernel {
    public:
        DiscontinuityKernel(kernel::KernelBuilder & b,
                StreamSet * const inputStreams,
                const unsigned int& threshold,
                StreamSet * const markStream);
    protected:
        void generatePabloMethod() override;
    
    private:
        unsigned int threshold;
    };

    class Stereo2MonoPabloKernel final : public PabloKernel {
    public:
        Stereo2MonoPabloKernel(kernel::KernelBuilder & b,
                StreamSet * const firstInputStreams,
                StreamSet * const secondInputStreams,
                StreamSet * const outputStreams);
    protected:
        void generatePabloMethod() override;
    };

    class AmplifyPabloKernel final : public PabloKernel {
    public:
        AmplifyPabloKernel(kernel::KernelBuilder & b,
                const unsigned int bitsPerSample,
                StreamSet * const inputStreams,
                const unsigned int& factor,
                StreamSet * const outputStreams);
    protected:
        void generatePabloMethod() override;

    private:
        unsigned int bitsPerSample;
        unsigned int numInputStreams;
        unsigned int factor;
    };

    class ConcatenateKernel final : public PabloKernel {
    public:
        ConcatenateKernel(kernel::KernelBuilder & b,
                StreamSet *const firstInputStreams,
                StreamSet *const secondInputStreams,
                StreamSet * const outputStreams);
    protected:
        void generatePabloMethod() override;

    private:
        unsigned int numFirstInputStreams;
        unsigned int numSecondInputStreams;
    };
}
