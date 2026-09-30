# microbiome
Computational Microbiome Analysis Workshop – Final Project Instructions
Project description
You will work with a large-scale multi-omic dataset (MetaCardis cohort), which includes
microbiome and metabolomic data across multiple populations. A key challenge in this project
is that some omics modalities may be missing for certain samples, and your goal is to develop
a computational pipeline that can accurately estimate distances between samples, based on
microbiome and metabolome data combined, despite this missing information.
A naïve approach would involve imputing the missing data (e.g. using KNN, Random Forest, or
simple statistics such as mean/median) and then computing distances. However, your task is
to go beyond naïve solutions and develop a more robust and well-justified approach.
The project will be done in groups of 2 students, though groups of 1 or 3 may be approved
under special circumstances. If you are looking for a partner, you may use the relevant forum
on Moodle.
Milestones
Throughout the course, there will be three milestones to ensure that progress is being made
and to provide a formal opportunity for group discussions. Prior to each milestone meeting,
you need to prepare a short presentation (10-20 minutes) that summarizes your progress and
future plans. meetings will be coordinated via email. Please send your PowerPoint presentation
at least 24 hours in advance.
Minimum progression expected and deadlines for each milestone is as follows:
Deadline Minimum progression expected
Milestone 1 1/7/26 Present ideas for the ML pipeline (preprocessing, ML
models, evaluation plan), including an overview of the
computational method or a concept from a paper or a
blog post that inspires your idea.
Present meaningful and relevant exploration results
and identify the key challenges in the data.
Naïve modeling (mandatory baseline): Predict missing
omics data using methods such as KNN or Random
Forest
Milestone 2 10/9/26 Presenting the pipeline, preliminary pipeline results and
comparison to the naïve model you showed in
milestone 1 (performance and running time). Test How
the missing data percentage effects the pipeline
performance.
Milestone 3 15/10/25 Presenting the final results, including the pipeline,
validation, and possible future directions. You will
receive two additional datasets beforehand. Model
predictions on the main MetaCardis dataset and the
additional datasets must be sent at least 3 days in
advance. During the meeting, we will present your
pipeline results on our data.
Guiding questions for data exploration in milestone 1
You will use the MetaCardis cohort, a large-scale multi-omic study aiming to identify
biomarkers for metabolic and cardiovascular diseases. The data includes metadata, stool
metagenomic data (species-level with CLR transformation), and metabolomic data. See "data
description" file in Moodle for more information.
For milestone 1:
• Explore the data provided and identify any problems you think you need to handle in
your pipeline. Below are questions meant to serve as a starting point for data
exploration, and you are not required to answer all of them. Be sure to explore
additional questions that could further help you develop your pipeline. In milestone 1
present only the exploration results that are relevant to your pipeline and decisions.
o What is the distribution of diseases in the metadata, and how do these
categories compare in terms of sample size and demographics? Which columns
contain missing values?
o How do microbiome composition profiles differ between disease groups and
healthy controls?
o Calculate distances between all microbiome samples, plot PCoA and calculate
average distance between groups.
o Could you detect specific bacteria that are differentially abundant in cases
compared to control?
o What are the key trends in the metabolome data across different disease
states? Are certain metabolites consistently elevated or reduced?
o How are the microbiome and metabolome data correlated? Which microbial
taxa show strong associations with specific metabolites?
o How might confounding factors (such as age and gender) influence the
observed patterns in the microbiome-metabolome data? Do they correlate with
both microbial diversity and metabolomic profiles
• Naïve modeling: Establish a benchmark by predicting missing omics data using KNearest Neighbors (KNN) or Random Forest, or by using a naive statistic (average,
median, etc.) to impute data and evaluate the performance.
o Evaluate how good is predicting disease state with\without imputation?
• Review a computational method or a concept (not necessarily developed for
microbiome or even biological data) from a paper or blog post, and explain how it, or
an adapted version of it, could be applied to this task, including the reasoning behind
its suitability. The method may be applied at any stage of the pipeline, including feature
selection, representation learning, data modeling, prediction, or any other stage. This
method Refer to Moodle for suggested options and guiding questions.
Final submission guidelines
Final submission includes: 1. Report 2. GitHub with your code 3. CSV with predictions on test
set.
Report
A detailed pdf document describing the problem, the pipeline, validation and validation
results and possible future directions. The paper should be 4-5 pages long at most with up to
4 figures (not including bibliography). It should have the following section:
• Introduction – describing the problem, and a general description of the solution's
approach, pipeline and a summary of the findings.
• Methods – detailed description of the pipeline and validation.
• Results – detailed description of your findings and the validation results. You may
include the results of experiments you have done even if the results were not so
good, and you excluded them later.
• Discussion – summary of the results, their importance and future possible directions.
• Bibliography – we recommend using either Mendeley or Zotero to insert bibliography.
Creating figures to visualize the results, pipelines, etc. is strongly encouraged. To easily create
figures in Python we suggest using either seaborn, plotly or other similar Python libraries. The
report should be sent a week after milestone 3 meeting.
Code
Code should be submitted in GitHub. Keep the code clean, well-documented and easy to read.
Predictions on test data
Your pipeline must predict and output an 𝑛𝑋𝑛 distance matrix. For evaluation, we will
construct a reference ("ground truth") distance matrix by normalizing each microbiome and
metabolome features (including the left out data) using a standard scaler, perform PCA on the
combined normalized data, and compute the Euclidean distances between samples in the first
two principal components. We will evaluate your work by using a Mantel test to compare your
predicted distance matrix to our ground truth distance matrix.
Before the third milestone, you will receive two additional datasets. You are not required to
modify your pipeline based on these data; they are intended solely for validation purposes.
Predictions should be sent three days before milestone 3 meeting. During the meeting we will
present to you the model performance on the test set.
Grading
The final grade will be determined based on the model's performance, milestone meetings,
and the overall quality of the final project.
Creativity and research effort will be heavily rewarded, with grades of 96-100 reserved for
exceptional effort, innovative ideas, and particularly rigorous research. Creativity is preferred
over pure performance; a highly creative and well-justified approach can be valued more than
a marginal performance improvement. Ensure that your modifications and ideas are
meaningful and well-reasoned, not arbitrary.
