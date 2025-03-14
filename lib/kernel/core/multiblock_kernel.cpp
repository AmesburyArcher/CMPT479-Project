/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#include <kernel/core/kernel.h>
#include <kernel/core/streamset.h>
#include <toolchain/toolchain.h>
#include <kernel/core/kernel_builder.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Transforms/Utils/Local.h>
#include <sstream>

using namespace llvm;

namespace kernel {

using PortType = Kernel::PortType;
using StreamSetPort = Kernel::StreamSetPort;
using AttrId = Attribute::KindId;
using RateId = ProcessingRate::KindId;
using Rational = ProcessingRate::Rational;

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateKernelMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void MultiBlockKernel::generateKernelMethod(KernelBuilder & b) {
    generateMultiBlockLogic(b, b.getNumOfStrides());
}

// MULTI-BLOCK KERNEL CONSTRUCTOR
MultiBlockKernel::MultiBlockKernel(LLVMTypeSystemInterface & ts,
    std::string && kernelName,
    Bindings && stream_inputs,
    Bindings && stream_outputs,
    Bindings && scalar_parameters,
    Bindings && scalar_outputs,
    InternalScalars && internal_scalars)
: MultiBlockKernel(ts,
    TypeId::MultiBlock,
    std::move(kernelName),
    std::move(stream_inputs),
    std::move(stream_outputs),
    std::move(scalar_parameters),
    std::move(scalar_outputs),
    std::move(internal_scalars)) {

}

MultiBlockKernel::MultiBlockKernel(LLVMTypeSystemInterface & ts,
    const TypeId typeId,
    std::string && kernelName,
    Bindings && stream_inputs,
    Bindings && stream_outputs,
    Bindings && scalar_parameters,
    Bindings && scalar_outputs,
    InternalScalars && internal_scalars)
: Kernel(ts, typeId,
     std::move(kernelName),
     std::move(stream_inputs),
     std::move(stream_outputs),
     std::move(scalar_parameters),
     std::move(scalar_outputs),
     std::move(internal_scalars)) {

}

}
