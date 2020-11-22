# fair-truth-discovery

This is the code for paper titled "Towards Fair Truth Discovery from Biased Crowdsourced Answers" for SIGKDD'20 submission. Would be glad if it helps, and if it does please do us a favor and cite the following paper:

```
@inproceedings{li2020towards,
  title={Towards Fair Truth Discovery from Biased Crowdsourced Answers},
  author={Li, Yanying and Sun, Haipei and Wang, Wendy Hui},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={599--607},
  year={2020}
}
```

1. Data

	- Crowd Judgement dataset:
		
		* This dataset comes from [here](https://farid.berkeley.edu/downloads/publications/scienceadvances17/)

		* The data is processed and can be read by file file "data_reader.py", just import it and call "data_crowdjudgement(A)", where A is the parameter from ["race"|"gender"]. This function will return three lists/matrices: answer, truth and workerid.

	- Synthetic dataset:
		* Synthetic dataset is generated following the setting in our paper, it can be read by calling "data_synthetic()" in "data_reader.py". The function returns workers' answers and ground truth.

2. Approaches

	- Pre-processing
		* To use pre-processing, import "pre_processing.py" and call "FairTD_Pre(answer,theta)", where "answer" is the answer matrix and "theta" is the user-specified threshold.

	- In-processing
		* To use in-processing, import "in_processing.py" and call "FairTD_In(answer,theta)", where "answer" is the answer matrix and "theta" is the user-specified threshold.

	- Post-processing
		* Post-processing is designed by combining one of three basic truth discovery algorithms with massaging approach:  
			(1) Majority Voting;  
			(2) EM Algorithm;  
			(3) CATD in paper "A confidence-aware approach for truth discovery on long-tail data" by Li et al.  

		* The massaging approaches comes from paper "Data preprocessing techniques for classification without discrimination" by Kamiran et al.

		* There all three approaches are in file "post_processing.py". The names of functions are:  
			(1) FairTD_Post_MV(answer,theta)  
			(2) FairTD_Post_EM(answer,theta)  
			(3) FairTD_Post_CATD(answer,theta)  

3. Tools

	- The class "Metric" in "metrics.py" is written to measure accuracy and disparity when ground truth are given. 