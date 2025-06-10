# Stochastic Extragradient with Flip-Flop Shuffling & Anchoring: Provable Improvements

These are the Python codes that are used in the experiments in our paper *Stochastic Extragradient with Flip-Flop Shuffling & Anchoring: Provable Improvements* (NeurIPS 2024, [arXiv](https://arxiv.org/abs/2501.00511)). 

To reproduce the plots for the experiments on the monotone setting(i.e., the experiment in Section 6 and Appendix I.2), simply run the file **SEG-monotone.py**. You won't need to change anything written in the code. A file named **monotone_results.pdf** will be generated. 

To reproduce the plot for the experiment in Appendix I.3 which compares our SEG-FFA to the method by Hsieh et al. (2020), simply run the file **SEG-Hsieh.py**. You won't need to change anything written in the code. A file named **geom_mean_hsieh.pdf** will be generated. 

In the code for the experiments on the strongly monotone case, the setting mentioned in Section 6 is set as the default. Thus, simply running the file **SEG-strongly_monontone.py** will generate the plot that is shown in Section 6, which is also the plot in Figure 5(d). To generate other plots in Figure 5, please change the values of the parameters **T** and **eta_k** in lines 10 and 11 of the code before you run it. A file named **geom_mean_scsc.pdf** will be generated. 
