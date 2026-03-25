add_executable(test_load test_load.cpp)
target_link_libraries(test_load PRIVATE sam3)

add_executable(test_tokenizer test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE sam3)

add_executable(test_vit test_vit.cpp)
target_link_libraries(test_vit PRIVATE sam3)

add_executable(test_phase5 test_phase5.cpp)
target_link_libraries(test_phase5 PRIVATE sam3)

add_executable(test_phase3 test_phase3.cpp)
target_link_libraries(test_phase3 PRIVATE sam3)

add_executable(test_phase4 test_phase4.cpp)
target_link_libraries(test_phase4 PRIVATE sam3)
