/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/unicode/UCD_property_kernel.h>

#include <kernel/core/kernel.h>
#include <re/adt/re_name.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <unicode/data/PropertyObjects.h>
#include <unicode/utf/utf_compiler.h>
#include <kernel/core/kernel_builder.h>
#include <pablo/builder.hpp>
#include <pablo/pe_zeroes.h>
#include <llvm/Support/ErrorHandling.h>

using namespace kernel;
using namespace pablo;
using namespace cc;


UnicodePropertyKernelBuilder::UnicodePropertyKernelBuilder(LLVMTypeSystemInterface & ts, re::Name * property_value_name, StreamSet * Source, StreamSet * property, pablo::BitMovementMode mode)
: UnicodePropertyKernelBuilder(ts, property_value_name, Source, property, mode, [&]() -> std::string {
    return std::to_string(Source->getNumElements()) +
           "x" + std::to_string(Source->getFieldWidth()) +
            property_value_name->getFullName() +
            UTF::kernelAnnotation() +
            pablo::BitMovementMode_string(mode);
}()) {

}

UnicodePropertyKernelBuilder::UnicodePropertyKernelBuilder(LLVMTypeSystemInterface & ts, re::Name * property_value_name, StreamSet * Source, StreamSet * property, pablo::BitMovementMode mode, std::string && propValueName)
: PabloKernel(ts,
"UCD:" + getStringHash(propValueName) ,
{Binding{"source", Source, FixedRate(1), LookAhead(3)}},
{Binding{"property_stream", property}})
, mPropNameValue(propValueName)
, mName(property_value_name)
, mBitMovement(mode) {
}

llvm::StringRef UnicodePropertyKernelBuilder::getSignature() const {
    return llvm::StringRef{mPropNameValue};
}

void UnicodePropertyKernelBuilder::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    UTF::UTF_Compiler unicodeCompiler(getInput(0), pb, mBitMovement);
    pablo::Var * propertyVar = pb.createVar(mName->getFullName(), pb.createZeroes());
    re::RE * property_defn = mName->getDefinition();
    if (re::CC * propertyCC = llvm::dyn_cast<re::CC>(property_defn)) {
        //unicodeCompiler.addTarget(propertyVar, propertyCC);
        unicodeCompiler.compile({propertyVar}, {propertyCC});
    } else if (re::PropertyExpression * pe = llvm::dyn_cast<re::PropertyExpression>(property_defn)) {
        if (pe->getKind() == re::PropertyExpression::Kind::Codepoint) {
            re::CC * propertyCC = llvm::cast<re::CC>(pe->getResolvedRE());
            //unicodeCompiler.addTarget(propertyVar, propertyCC);
            unicodeCompiler.compile({propertyVar}, {propertyCC});
        }
    }
    Var * const property_stream = getOutputStreamVar("property_stream");
    pb.createAssign(pb.createExtract(property_stream, pb.getInteger(0)), propertyVar);
}

UnicodePropertyBasis::UnicodePropertyBasis(LLVMTypeSystemInterface & ts, UCD::EnumeratedPropertyObject * enumObj, StreamSet * Source, StreamSet * PropertyBasis)
: PabloKernel(ts,
"UCD:" + getPropertyFullName(enumObj->getPropertyCode()) + "_basis",
{Binding{"source", Source}},
{Binding{"property_basis", PropertyBasis}})
, mEnumObj(enumObj) {

}

void UnicodePropertyBasis::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    UTF::UTF_Compiler unicodeCompiler(getInput(0), pb);
    std::vector<UCD::UnicodeSet> & bases = mEnumObj->GetEnumerationBasisSets();
    std::vector<Var *> targetVars(bases.size());
    std::vector<re::CC *> basisCCs(bases.size());
    for (unsigned i = 0; i < bases.size(); i++) {
        std::string vname = "basis" + std::to_string(i);
        targetVars[i] = pb.createVar(vname, pb.createZeroes());
        basisCCs[i] = re::makeCC(bases[i], &Unicode);
    }
    unicodeCompiler.compile(targetVars, basisCCs);
    Var * const property_basis = getOutputStreamVar("property_basis");
    for (unsigned i = 0; i < bases.size(); i++) {
        pb.createAssign(pb.createExtract(property_basis, pb.getInteger(i)), targetVars[i]);
    }
}
