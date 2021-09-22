#include "post_process.h"

#include <cassert>
#include <cstdlib>
#include <string>
#include <sstream>
#include <llvm/Support/Compiler.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/ADT/SmallVector.h>

/*
 * <Json> ::= <Object>
 *          | <Array>
 * 
 * <Object> ::= '{' '}'
 *            | '{' <Members> '}'
 * 
 * <Members> ::= <Pair>
 *             | <Pair> ',' <Members>
 * 
 * <Pair> ::= String ':' <Value>
 * 
 * <Array> ::= '[' ']'
 *           | '[' <Elements> ']'
 * 
 * <Elements> ::= <Value>
 *              | <Value> ',' <Elements>
 * 
 * <Value> ::= String
 *           | Number
 *           | <Object>
 *           | <Array>
 *           | true
 *           | false
 *           | null
 */

enum JSONState {
    JInit = 0,
    JKStrBegin,
    JKStrEnd,
    JVStrBegin,
    JVStrEnd,
    JObjInit,
    JObjColon,
    JArrInit,
    JValue,
    JNextComma,
    JDone
};

static llvm::SmallVector<const u_int8_t *, 32> stack{};
static JSONState currentState = JInit;

static ptrdiff_t postproc_getColumn(const uint8_t * ptr, const uint8_t * lineBegin) {
    ptrdiff_t column = ptr - lineBegin;
    assert (column >= 0);
    return column;
}

static std::string postproc_getLineAndColumnInfo(const std::string str, const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum) {
    ptrdiff_t column = postproc_getColumn(ptr, lineBegin);
    std::stringstream ss;
    ss << str << " at line " << lineNum << " column " << column << " starting in:\n\n" << ptr;
    return ss.str();
}

static std::string postproc_getCustomLineAndColumnInfo(
    const std::string str,
    const std::string butStr,
    const uint8_t * ptr,
    const uint8_t * lineBegin,
    uint64_t lineNum)
{
    ptrdiff_t column = postproc_getColumn(ptr, lineBegin);
    std::stringstream ss;
    ss << "Found" << str << " at line " << lineNum << " column " << column << " " << butStr;
    return ss.str();
}

static bool isControl(const uint8_t * ptr) {
    if (*ptr == '}' || *ptr == ']' || *ptr == ',' || *ptr == ':') {
        return true;
    } else {
        return false;
    }
}

static void postproc_popAndFindNewState() {
    assert(!stack.empty());
    stack.pop_back();
    if (stack.empty()) {
        currentState = JDone;
        return;
    }
    const uint8_t last = *stack.back();
    if (last == '{' || last == '[') {
        currentState = JNextComma;
    } else {
        llvm_unreachable("The stack has an unknown char");
    }
}

static void postproc_parseArrOrObj(const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (*ptr == '{') {
        currentState = JObjInit;
    } else if (*ptr == '[') {
        currentState = JArrInit;
    } else {
        llvm::report_fatal_error(postproc_getLineAndColumnInfo("Error parsing object or array", ptr, lineBegin, lineNum));
    }
    stack.push_back(ptr);
}

static void postproc_parseStrOrPop(bool popAllowed, const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (*ptr == '"') {
        currentState = JKStrBegin;
    } else if (*ptr == '}' && popAllowed) {
        postproc_popAndFindNewState();
    } else {
        llvm::report_fatal_error(postproc_getLineAndColumnInfo("Error parsing object", ptr, lineBegin, lineNum));
    }
}

static bool postproc_parseValue(bool strict, const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (*ptr == '"') {
        currentState = JVStrBegin;
        return true;
    } else if (*ptr == '{' || *ptr == '[') {
        postproc_parseArrOrObj(ptr, lineBegin, lineNum, position);
        return true;
    } else if (!isControl(ptr)) {
        currentState = JValue;
        return true;
    } else if (strict) {
        llvm::report_fatal_error(postproc_getLineAndColumnInfo("Error parsing value", ptr, lineBegin, lineNum));
    }
    return false;
}

static void postproc_parseValueOrPop(bool popAllowed, const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (!postproc_parseValue(false, ptr, lineBegin, lineNum, position)) {
        if (*ptr == ']' && popAllowed) {
            postproc_popAndFindNewState();
        } else {
            llvm::report_fatal_error(postproc_getLineAndColumnInfo("Error parsing array", ptr, lineBegin, lineNum));
        }
    }
}

static void postproc_parseStr(const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (*ptr == '"' && currentState == JKStrBegin) {
        currentState = JKStrEnd;
    } else if (*ptr == '"' && currentState == JVStrBegin) {
        currentState = JVStrEnd;
    } else {
        llvm::report_fatal_error(postproc_getLineAndColumnInfo("Error parsing string", ptr, lineBegin, lineNum));
    }
}

static void postproc_parseColon(const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (*ptr == ':') {
        currentState = JObjColon;
    } else {
        llvm::report_fatal_error(postproc_getLineAndColumnInfo("Error parsing object", ptr, lineBegin, lineNum));
    }
}

static void postproc_parseCommaOrPop(const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (stack.empty()) {
        llvm::report_fatal_error(postproc_getLineAndColumnInfo("Stack is empty", ptr, lineBegin, lineNum));
        return;
    }
    const uint8_t last = *stack.back();
    if (*ptr == ',') {
        if (last == '{') {
            currentState = JObjInit;
        } else if (last == '[') {
            currentState = JArrInit;
        } else {
            llvm::report_fatal_error(postproc_getLineAndColumnInfo("Wrong char in stack", ptr, lineBegin, lineNum));
        }
    } else if (*ptr == '}' && last == '{') {
        postproc_popAndFindNewState();
    } else if (*ptr == ']' && last == '[') {
        postproc_popAndFindNewState();
    } else {
        llvm::report_fatal_error(postproc_getLineAndColumnInfo("Error parsing object", ptr, lineBegin, lineNum));
    }
}

static void postproc_eof(const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (stack.empty()) {
        currentState = JDone;
	return;
    }
    llvm::report_fatal_error(postproc_getCustomLineAndColumnInfo("EOF", "but the JSON is missing elements", ptr, lineBegin, lineNum));
}

void postproc_validateObjectsAndArrays(const uint8_t * ptr, const uint8_t * lineBegin, const uint8_t * /*lineEnd*/, uint64_t lineNum, uint64_t position) {
printf("%c\n", *ptr);
    	if (currentState == JInit) {
        postproc_parseArrOrObj(ptr, lineBegin, lineNum, position);
    } else if (currentState == JObjInit) {
        postproc_parseStrOrPop(true, ptr, lineBegin, lineNum, position);
    } else if (currentState == JArrInit) {
        postproc_parseValueOrPop(true, ptr, lineBegin, lineNum, position);
    } else if (currentState == JKStrBegin || currentState == JVStrBegin) {
        postproc_parseStr(ptr, lineBegin, lineNum, position);
    } else if (currentState == JKStrEnd) {
        postproc_parseColon(ptr, lineBegin, lineNum, position);
    } else if (currentState == JObjColon) {
        postproc_parseValue(true, ptr, lineBegin, lineNum, position);
    } else if (currentState == JVStrEnd || currentState == JValue || currentState == JNextComma) {
        postproc_parseCommaOrPop(ptr, lineBegin, lineNum, position);
    } else if (*ptr == EOF) {
        postproc_eof(ptr, lineBegin, lineNum, position);
    } else if (currentState == JDone) {
        llvm::report_fatal_error(postproc_getLineAndColumnInfo("JSON has been already processed", ptr, lineBegin, lineNum));
    }
}

void postproc_errorStreamsCallback(const uint8_t * ptr, const uint8_t * lineBegin, const uint8_t * /*lineEnd*/, uint64_t lineNum, uint8_t code) {
    const uint8_t KEYWORD_CODE = 0x1;
    const uint8_t UTF8_CODE = 0x2;
    const uint8_t NUMBER_CODE = 0x4;
    const uint8_t EXTRA_CODE = 0x8;
    if (*ptr == '\n' || *ptr == '\r') {
        return;
    }

    if (code == KEYWORD_CODE) {
        fprintf(stderr, "%s", postproc_getLineAndColumnInfo("illegal keyword", ptr, lineBegin, lineNum).c_str());
    } else if (code == UTF8_CODE) {
        fprintf(stderr, "%s", postproc_getLineAndColumnInfo("illegal utf8 char", ptr, lineBegin, lineNum).c_str());
    } else if (code == NUMBER_CODE) {
        fprintf(stderr, "%s", postproc_getLineAndColumnInfo("illegal number", ptr, lineBegin, lineNum).c_str());
    } else if (code == EXTRA_CODE) {
        fprintf(stderr, "%s", postproc_getLineAndColumnInfo("illegal char in json", ptr, lineBegin, lineNum).c_str());
    }
    fprintf(stderr, "\n");
}
