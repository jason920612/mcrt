# Converts a SPIR-V binary into a header with a uint32_t array.
# Usage: cmake -DINPUT=x.spv -DOUTPUT=x.spv.h -DSYMBOL=kName -P embed_spirv.cmake
file(READ ${INPUT} hex HEX)
string(LENGTH "${hex}" hex_len)
math(EXPR byte_count "${hex_len} / 2")
math(EXPR rem "${byte_count} % 4")
if(NOT rem EQUAL 0)
    message(FATAL_ERROR "${INPUT} is not a multiple of 4 bytes")
endif()

# Regroup little-endian bytes into 32-bit words.
string(REGEX REPLACE "(..)(..)(..)(..)" "0x\\4\\3\\2\\1u," words "${hex}")
string(REGEX REPLACE "((0x[0-9a-f]+u,){8})" "\\1\n    " words "${words}")

file(WRITE ${OUTPUT} "#pragma once\n#include <cstdint>\ninline constexpr uint32_t ${SYMBOL}[] = {\n    ${words}\n};\n")
