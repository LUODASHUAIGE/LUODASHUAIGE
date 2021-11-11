### This repo is created for saving all homework of STATS 507 Python in data science  👋
- 🔭 Each homework has got a .ipynb format and .py format. 
- 🌱 A special file is called PS2 Q3.ipynb, this file is for HW6 of STATS 507. The purpose of this file is to use Pandas to read, clean, and append several data files from the National Health and Nutrition Examination Survey NHANES. Four cohorts spanning the years 2011-2018 are chosen. The amount work is dosed by three parts (a),(b) and (c).
- (a) : I use Python and Pandas to read and append the demographic datasets keeping only columns containing the unique ids (SEQN), age (RIDAGEYR), race and ethnicity (RIDRETH3), education (DMDEDUC2), and marital status (DMDMARTL), along with the following variables related to the survey weighting: (RIDSTATR, SDMVPSU, SDMVSTRA, WTMEC2YR, WTINT2YR). Then I add an additional column identifying to which cohort each case belongs. Next I rename the columns with literate variable names using all lower case and convert each column to an appropriate type. Finally, I save the resulting data frame to a serialized “round-trip” format of your choosing (e.g. pickle, feather, or parquet).
- (b) : Repeat part a for the oral health and dentition data (OHXDEN_*.XPT) retaining the following variables: SEQN, OHDDESTS, tooth counts (OHXxxTC), and coronal cavities (OHXxxCTC).
- (c) : I report the number of cases there are in the two datasets above. 
- **LINK of this special file: .github/workflows/PS2 Q3.ipynb** 

