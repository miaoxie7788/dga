# Task name: DGA detection

Data science is becoming increasingly important in cybersecurity. So, in
this task, we ask you to apply your data science knowledge to help
detect the presence of malicious software.


## Context:

Once malicious software has been installed on a system it needs to be
told what to do. To receive instructions the malware needs to
communicate with a command and control server. Historically, the domain
name of the command and control server was hardcoded into the code of
the malware. This made it very easy for cybersecurity professionals to
block this communication channel by blacklisting the domain once they
discovered it.

To avoid this weakness malware programs have started to use Domain
Generating Algorithms (or DGAs). These algorithms generate a number of
random domain names where one of the generated domains is the control
server. The infected computed scans through these domains trying to
contact each, eventually it will try the correct domain. At this stage,
the malware can receive instructions remotely.

## Data:

The data for this task is available on S3:

https://datasci-test.s3-ap-southeast-2.amazonaws.com/dga_domains.csv

This dataset contains ~130,000 domains names each of which are labelled
"legit" or "dga".

## Objective:

Using the dataset provided, build a model which identifies domains that
are generated from DGAs. You are also welcome to treat this as a
multi-class classification problem and hence build a model to predict
the DGA sub-classes, but this is not required.

Please take on the mindset that you are delivering a product to a
client. With this in mind, consider what would be expected in this
situation.

## Implementation requirements:

Write modular, clean, and well-documented code. Include a command-line
interface where the expected input is a single domain name. Write both
unit and integration tests.


## Data science requirements:

To properly assess this task we need to be able to understand your
thought process. Accordingly, please explain any modelling/technical
decisions you made, for example,

Why have you chosen that particular model?
What steps have you taken to explore the data?
What metric(s) are you using to measure the performance and why?
How have you validated the model's performance?







