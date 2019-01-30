# Video Channel Recommendation
Since the influence of videos became essential in various fields worldwide, personalized video recommendation is considered a valuable task. 
Many real-world video platforms such as YouTube and Kuaishou provide personalized recommendations for both video items and video channels. 
This suggests that there is also a demand for channel recommendation. 
However, despite this demand, most of the academic research has focused only on video item recommendation while leaving the channel recommendation problem virtually unaddressed. 
To address this problem, we define a new channel recommendation task and introduce a novel recommendation model for the task. 
Our proposed model learns the representations for user preferences from user-channel subscription information and the representations for video channels using the available title and category information of the videos from the channels. 
To validate our model, we constructed a largescale video channel subscription dataset with data collected from YouTube. 
The experimental results show that our model outperforms the state-of-the-art recommender systems that are proven to be effective in different recommendation problems. 
To the best of our knowledge, this is the first academic work to propose a video channel recommendation model. 
Moreover, our proposed channel recommendation task and the existing video item recommendation tasks are mutually beneficial. 
Learned representations for user preferences from subscription information can be used for downstream application of video item recommendation. 
We make our dataset and our code publicly available at https://github.com/sigir2019-vcr.
We hope that our newly defined channel recommendation task and constructed dataset can be useful community resources for future recommendation research.

## Getting Started
The experiments were conducted on a single TITAN Xp GPU machine which has 12GB of RAM.
The implemntation code of VCR was tested with the following requirements:
### Prerequisites
*   **[`Python2.7.12`](https://www.python.org/downloads/release/python-2712/)**
*   **[`PyTorch0.4.1`](https://pytorch.org/get-started/previous-versions/)**

### Usage
Following command runs training CAPRE with default hyper-paramters.

```bash
git clone https://github.com/sigir2019-vcr/VCR
cd VCR
python train.py
```

If you want to change the hyper-parameter settings, please see the `parameters.py` file.

## Citation

If we submit the paper to a conference, we will update the BibTeX.

## Contact information

For help or issues using VCR, please submit a GitHub issue. Please contact Annoymized (`annonymized`) for communication related to VCR.
