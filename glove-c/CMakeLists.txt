project(glove)

set(CMAKE_CXX_STANDARD 14)

add_executable(cmake_cooccur src/cooccur.c)
add_executable(cmake_glove src/glove.c)
add_executable(cmake_shuffle src/shuffle.c)
add_executable(cmake_vocab_count src/vocab_count.c)