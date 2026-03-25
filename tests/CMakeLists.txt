add_executable(test_load test_load.cpp)
target_link_libraries(test_load PRIVATE sam3)

add_executable(test_tokenizer test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE sam3)
