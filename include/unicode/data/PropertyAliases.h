#ifndef PROPERTYALIASES_H
#define PROPERTYALIASES_H
/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 *  2025
 *
 *  This file is generated by UCD_properties.py - manual edits may be lost.
 */

#include <string>

namespace UCD {
    enum property_t : int {
        identity, cjkAccountingNumeric, cjkOtherNumeric, cjkPrimaryNumeric,
        nv, bmg, bpb, cf, cjkCompatibilityVariant, dm, EqUIdeo, FC_NFKC, lc,
        NFKC_CF, NFKC_SCF, scf, slc, stc, suc, tc, uc, cjkIICore,
        cjkIRG_GSource, cjkIRG_HSource, cjkIRG_JSource, cjkIRG_KPSource,
        cjkIRG_KSource, cjkIRG_MSource, cjkIRG_SSource, cjkIRG_TSource,
        cjkIRG_UKSource, cjkIRG_USource, cjkIRG_VSource, cjkRSUnicode, isc,
        JSN, kEH_Cat, kEH_Desc, kEH_HG, kEH_IFAO, kEH_JSesh, na, na1,
        Name_Alias, scx, age, blk, sc, bc, bpt, ccc, dt, ea, gc, GCB, hst,
        InCB, InPC, InSC, jg, jt, lb, NFC_QC, NFD_QC, NFKC_QC, NFKD_QC, nt,
        SB, vo, WB, AHex, Alpha, Bidi_C, Bidi_M, Cased, CE, CI, Comp_Ex,
        CWCF, CWCM, CWKCF, CWL, CWT, CWU, Dash, Dep, DI, Dia, EBase, EComp,
        EMod, Emoji, EPres, Ext, ExtPict, Gr_Base, Gr_Ext, Gr_Link, Hex,
        Hyphen, ID_Compat_Math_Continue, ID_Compat_Math_Start, IDC, Ideo,
        IDS, IDSB, IDST, IDSU, Join_C, kEH_NoMirror, kEH_NoRotate, LOE,
        Lower, Math, MCM, NChar, OAlpha, ODI, OGr_Ext, OIDC, OIDS, OLower,
        OMath, OUpper, Pat_Syn, Pat_WS, PCM, QMark, Radical, RI, SD, STerm,
        Term, UIdeo, Upper, VS, WSpace, XIDC, XIDS, XO_NFC, XO_NFD, XO_NFKC,
        XO_NFKD, emoji, emojipresentation, emojimodifier, emojimodifierbase,
        emojicomponent, extendedpictographic, alnum, xdigit, blank, print,
        word, graph, g, w,
        Undefined = -1};
    const std::string & getPropertyEnumName(const property_t);
    const std::string & getPropertyFullName(const property_t);
    property_t getPropertyCode(std::string & propertyIdent);
}

#endif
