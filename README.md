# fair-truth-discovery

This is the code for paper titled "Towards Fair Truth Discovery from Biased Crowdsourced Answers" for SIGKDD'20 submission.

1. Data

	- Crowd Judgement dataset:
		
		* This dataset comes from https://farid.berkeley.edu/downloads/publications/scienceadvances17/
		* The data is processed and can be read by file file "data_reader.py", just import it and call "data_crowdjudgement(A)", where A is the parameter from ["race"|"gender"]. This function will return three lists/matrices: answer, truth and workerid.

	- Synthetic dataset:
		* Synthetic dataset is generated following the setting in our paper, it can be read by calling "data_synthetic()" in "data_reader.py". The function returns workers' answers and ground truth.

2. Approaches

	- Pre-processing
		* To be filled.

	- In-processing
		* To use in-processing, import "in_processing.py" and call "FairTD_In(answer,theta)", where "answer" is the answer matrix and "theta" is the user-specified threshold.

	- Post-processing
		* To be filled