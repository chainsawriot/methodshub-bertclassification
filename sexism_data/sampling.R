require(readr)
require(dplyr)

set.seed(42)
sexism_data <- read_csv(here::here("sexism_data", "sexism_data.csv"), col_types = readr::cols(.default = readr::col_character()))

sexism_data %>% filter(sexist == "True") %>% sample_n(100) -> sexist_tweets
sexism_data %>% filter(sexist == "False") %>% sample_n(100) -> nonsexist_tweets

write_csv(sample_frac(dplyr::bind_rows(sexist_tweets, nonsexist_tweets)), here::here("sexism_data", "sexism_sample.csv"))
