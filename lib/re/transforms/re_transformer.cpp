/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <re/transforms/re_transformer.h>

#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <re/adt/adt.h>
#include <re/toolchain/toolchain.h>

using namespace llvm;

namespace re {

RE * RE_Transformer::transformRE(RE * re, NameTransformationMode m) {
    mInitialRE = re;
    mNameTransform = m;
    RE * finalRE = transform(re);
    bool ShowRE = PrintOptionIsSet(ShowAllREs) && !mTransformationName.empty();
    if (PrintOptionIsSet(ShowREs) && (mInitialRE != finalRE)) {
        ShowRE |= !mTransformationName.empty() && (mTransformationName[0] != '.');
    }
    if (ShowRE)  {
        errs() << mTransformationName << ":\n";
        showProcessing();
        errs() << Printer_RE::PrintRE(finalRE) << '\n';
    }
    return finalRE;
}

void RE_Transformer::showProcessing() {
}

RE * RE_Transformer::transform(RE * const from) {
    assert (from);
    using T = RE::ClassTypeId;
    RE * to = from;
    #define TRANSFORM(Type) \
case T::Type: to = transform##Type(llvm::cast<Type>(from)); break
    switch (from->getClassTypeId()) {
        TRANSFORM(Alt);
        TRANSFORM(Any);
        TRANSFORM(Assertion);
        TRANSFORM(CC);
        TRANSFORM(Range);
        TRANSFORM(Diff);
        TRANSFORM(End);
        TRANSFORM(Intersect);
        TRANSFORM(Name);
        TRANSFORM(Capture);
        TRANSFORM(Reference);
        TRANSFORM(Group);
        TRANSFORM(Rep);
        TRANSFORM(Seq);
        TRANSFORM(Start);
        TRANSFORM(Permute);
        TRANSFORM(PropertyExpression);
        default: llvm_unreachable("Unknown RE type");
    }
    #undef TRANSFORM
    assert (to);
    return to;
}

RE * RE_Transformer::transformName(Name * nm) {
    if (mNameTransform == NameTransformationMode::None) {
        return nm;
    }
    RE * const defn = nm->getDefinition();
    if (LLVM_UNLIKELY(defn == nullptr)) {
        UndefinedNameError(nm);
    }
    RE * t = transform(defn);
    if (t == defn) return nm;
    return t; //makeName(nm->getFullName(), t);
}

RE * RE_Transformer::transformCapture(Capture * c) {
    RE * const captured = c->getCapturedRE();
    RE * t = transform(captured);
    if (t == captured) return c;
    return makeCapture(c->getName(), t);
}

RE * RE_Transformer::transformReference(Reference * r) {
    return r;
}

RE * RE_Transformer::transformAny(Any * a) {
    return a;
}

RE * RE_Transformer::transformCC(CC * cc) {
    return cc;
}

RE * RE_Transformer::transformStart(Start * s) {
    return s;
}

RE * RE_Transformer::transformEnd(End * e) {
    return e;
}

RE * RE_Transformer::transformSeq(Seq * seq) {
    SmallVector<RE *, 16> elems;
    elems.reserve(seq->size());
    bool any_changed = false;
    for (RE * e : *seq) {
        RE * e1 = transform(e);
        if (e1 != e) any_changed = true;
        elems.push_back(e1);
    }
    if (!any_changed) return seq;
    return makeSeq(elems.begin(), elems.end());
}

RE * RE_Transformer::transformAlt(Alt * alt) {
    SmallVector<RE *, 16> elems;
    elems.reserve(alt->size());
    bool any_changed = false;
    for (RE * e : *alt) {
        RE * e1 = transform(e);
        if (e1 != e) any_changed = true;
        elems.push_back(e1);
    }
    if (!any_changed) return alt;
    return makeAlt(elems.begin(), elems.end());
}

RE * RE_Transformer::transformRep(Rep * r) {
    RE * x0 = r->getRE();
    RE * x = transform(x0);
    if (x == x0) {
        return r;
    } else {
        return makeRep(x, r->getLB(), r->getUB());
    }
}

RE * RE_Transformer::transformIntersect(Intersect * ix) {
    RE * x0 = ix->getLH();
    RE * y0 = ix->getRH();
    RE * x = transform(x0);
    RE * y = transform(y0);
    if ((x == x0) && (y == y0)) {
        return ix;
    } else {
        return makeIntersect(x, y);
    }
}

RE * RE_Transformer::transformDiff(Diff * d) {
    RE * x0 = d->getLH();
    RE * y0 = d->getRH();
    RE * x = transform(x0);
    RE * y = transform(y0);
    if ((x == x0) && (y == y0)) {
        return d;
    } else {
        return makeDiff(x, y);
    }
}

RE * RE_Transformer::transformRange(Range * rg) {
    RE * x0 = rg->getLo();
    RE * y0 = rg->getHi();
    RE * x = transform(x0);
    RE * y = transform(y0);
    if ((x == x0) && (y == y0)) {
        return rg;
    } else {
        return makeRange(x, y);
    }
}

RE * RE_Transformer::transformGroup(Group * g) {
    RE * x0 = g->getRE();
    RE * x = transform(x0);
    if (x == x0) {
        return g;
    } else {
        return makeGroup(g->getMode(), x, g->getSense());
    }
}

RE * RE_Transformer::transformAssertion(Assertion * a) {
    RE * x0 = a->getAsserted();
    RE * x = transform(x0);
    if (x == x0) {
        return a;
    } else {
        return makeAssertion(x, a->getKind(), a->getSense());
    }
}

RE * RE_Transformer::transformPermute(Permute * p) {
    SmallVector<RE *, 16> elems;
    elems.reserve(p->size());
    bool any_changed = false;
    for (RE * e : *p) {
        RE * e1 = transform(e);
        if (e1 != e) any_changed = true;
        elems.push_back(e1);
    }
    if (!any_changed) return p;
    return makePermute(elems.begin(), elems.end());
}

RE * RE_Transformer::transformPropertyExpression(PropertyExpression * pe) {
    RE * const defn = pe->getResolvedRE();
    if (LLVM_UNLIKELY(defn == nullptr)) return pe;
    if ((mNameTransform == NameTransformationMode::TransformDefinition) ||
          isa<Reference>(defn)) {
        RE * t = transform(defn);
        if (t != defn) return t;
    }
    return pe;
}

} // namespace re
