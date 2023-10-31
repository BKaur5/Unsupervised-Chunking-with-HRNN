from library.heurestic_definition import double_word_heurestic, single_word_heurestic

TEST_DATA_PATH = "HRNNdata_en/test.pkl"
TEST_TAGS_PATH = "HRNNdata_en/test_tag.pkl"

print(single_word_heurestic(TEST_DATA_PATH,TEST_TAGS_PATH))
print(double_word_heurestic(TEST_DATA_PATH,TEST_TAGS_PATH))