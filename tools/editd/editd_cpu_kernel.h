/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <kernel/core/kernel.h>

namespace llvm { class Module; }

namespace IDISA { class IDISA_Builder; }

namespace kernel {

class editdCPUKernel : public BlockOrientedKernel {
public:

    editdCPUKernel(LLVMTypeSystemInterface & ts,
                   const unsigned editDistance, const unsigned patternLen, const unsigned groupSize,
                   Scalar * const pattStream,
                   StreamSet * const CCStream, StreamSet * const ResultStream);

protected:
    void generateDoBlockMethod(KernelBuilder & idb) override;
    void generateFinalBlockMethod(KernelBuilder & idb, llvm::Value * remainingBytes) override;
    void bitblock_advance_ci_co(KernelBuilder & idb, llvm::Value * val, unsigned shift, llvm::Value * stideCarryArr, unsigned carryIdx, std::vector<std::vector<llvm::Value *>> & adv, std::vector<std::vector<int>> & calculated, int i, int j) const;
    void reset_to_zero(std::vector<std::vector<int>> & calculated);
private:
    const unsigned mEditDistance;
    const unsigned mPatternLen;
    const unsigned mGroupSize;

};



}
