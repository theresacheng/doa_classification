# Get mean post-censored FD for DOA project
# T Cheng | 2020.01.23

## Load required packages ##
packages <-  c("dplyr", "tidyr", "ggplot2", "R.matlab", "rio")
if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(packages, rownames(installed.packages())))  
}
lapply(packages, library, character.only = TRUE)

working_dir <- "~/Box/projects/dim_of_adversity"
studies <- c("uo_tds", "MINA", "K_study")
study_names <- c("TDS", "MINA2", "K_study")
beh_data <- readRDS("/Users/theresacheng/projects/doa_classification/data/beh_data.rds")

# read _5_minutes_of_data_at_0.2_threshold.txt files, save mean FD and post-censored mean FD
mean_FD_df <- data.frame(subject_id = as.character(),
                      mean_FD = as.numeric(),
                      mean_FD_censored = as.numeric())

for (i in 1:length(studies)){

  # get a  list of the subject ids
  subjects <- beh_data %>% filter(study == study_names[i]) %>% select("sub") #list.dirs(path = paste0(working_dir, "/", studies[i], "/data"), full.names = FALSE, recursive = FALSE)
  subjects <- subjects$sub
  
  for (j in 1:length(subjects)){
    # read in files
    if (studies[i] == "uo_tds"){
        motion_numbers <- readMat(paste0(working_dir, "/", studies[i], "/data/", subjects[j], "/motion/motion_numbers.mat"))
        censor_vec <- import(paste0(working_dir, "/", studies[i], "/data/", subjects[j], "/motion/power_2014_FD_only.mat_0.2_cifti_censor_FD_vector_5_minutes_of_data_at_0.2_threshold.txt"))
    } else {
      motion_numbers <- readMat(paste0(working_dir, "/", studies[i], "/data/", subjects[j], "/motion_numbers.mat"))
      censor_vec <- import(paste0(working_dir, "/", studies[i], "/data/", subjects[j], "/power_2014_FD_only.mat_0.2_cifti_censor_FD_vector_5_minutes_of_data_at_0.2_threshold.txt"))
    }
    
    FD_df <- as.data.frame(motion_numbers$motion.numbers[7])
    colnames(FD_df) = "FD"
    censored_FD <- filter(FD_df, censor_vec == 1)
    
    temp_df <- data.frame(subject_id = subjects[j], 
                          mean_FD = mean(FD_df$FD),
                          mean_FD_censored = mean(censored_FD$FD))
    
    mean_FD_df <- rbind(mean_FD_df, temp_df) 
  }
}

## get extra files for the K_study that end in All_Good_Frames.txt
mean_FD_df_extra_K <- data.frame(subject_id = as.character(),
                         mean_FD = as.numeric(),
                         mean_FD_censored = as.numeric())

i = 3
subjects <- beh_data %>% filter(study == study_names[i]) %>% select("sub") #list.dirs(path = paste0(working_dir, "/", studies[i], "/data"), full.names = FALSE, recursive = FALSE)
subjects <- subjects$sub
  
for (j in 1:length(subjects)){
    motion_numbers <- readMat(paste0(working_dir, "/", studies[i], "/data/", subjects[j], "/motion_numbers.mat"))
    censor_vec <-  import(paste0(working_dir, "/", studies[i], "/data/", subjects[j], "/power_2014_FD_only.mat_0.2_cifti_censor_FD_vector_All_Good_Frames.txt"))
    
FD_df <- as.data.frame(motion_numbers$motion.numbers[7])
colnames(FD_df) = "FD"
censored_FD <- filter(FD_df, censor_vec == 1)

temp_df <- data.frame(subject_id = subjects[j], 
                      mean_FD = mean(FD_df$FD),
                      mean_FD_censored = mean(censored_FD$FD))

mean_FD_df_extra_K <- rbind(mean_FD_df_extra_K, temp_df)
}

# combine the two dataframes: mean_FD_df and mean_FD_df_extra_K
mean_FD_df_all <- rbind(mean_FD_df, mean_FD_df_extra_K)
mean_FD_df_all_no_duplicates <- distinct(mean_FD_df_all, subject_id, mean_FD, .keep_all = TRUE)

# pare down to beh_data subject list
colnames(mean_FD_df_all_no_duplicates)[1] <- "sub"
beh_data_w_mean_FD <- left_join(beh_data, mean_FD_df_all_no_duplicates)

#saveRDS(beh_data_w_mean_FD, file = "/Users/theresacheng/projects/doa_classification/data/beh_data.rds")

# do this for beh_TDS, MINA, and K_study
beh_data_TDS <- readRDS("/Users/theresacheng/projects/doa_classification/data/beh_data.rds") %>% 
  filter(study == "TDS")
saveRDS(beh_data_TDS, file = "/Users/theresacheng/projects/doa_classification/data/beh_TDS.rds")

beh_data_MINA <- readRDS("/Users/theresacheng/projects/doa_classification/data/beh_data.rds") %>% 
  filter(study == "MINA2")
saveRDS(beh_data_MINA, file = "/Users/theresacheng/projects/doa_classification/data/beh_MINA.rds")

beh_data_K <- readRDS("/Users/theresacheng/projects/doa_classification/data/beh_data.rds") %>% 
  filter(study == "K_study")
saveRDS(beh_data_K, file = "/Users/theresacheng/projects/doa_classification/data/beh_K_study.rds")


