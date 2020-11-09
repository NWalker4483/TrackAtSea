# AI Tracks at Sea

The purpose of this document is to give a quick overview of the tasks and assessment criteria for the [AI Tracks at Sea](https://urldefense.proofpoint.com/v2/url?u=https-3A__www.challenge.gov_challenge_AI-2Dtracks-2Dat-2Dsea_&d=DwMFaQ&c=0CCt47_3RbNABITTvFzZbA&r=dZo2HcoPtVrWSl5PTL0fWbxL8T2cOeHTSUoFQVj7Cd4&m=DatDTm7XdNK8lvuDPyscQhdQ183J69XjjY2FP8zumuU&s=anghwHCn8GmNXI331_rvYHSOT1DDGL5OWaDwxEkgr5o&e=) Challenge.

## AI Tracks at Sea Challenge

#### Description

Current traffic avoidance software relies on AIS and radar for tracking other craft and avoiding collisions. However, in a contested environment, emitting radar energy presents a vulnerability to detection by adversaries. Deactivating RF transmitting sources mitigates the threat of detection, but without additional sensing and sensor fusion, it would also degrade the USV’s ability to monitor shipping traffic in its vicinity.

The technology developed for this prize challenge will solve this problem by developing a computer vision system that will plot the tracks of shipping traffic exclusively using the passive sensing capability of onboard cameras. This technology would also be applicable for the Navy to document vessels performing unsafe navigation on the high seas.

#### Rules

Each Team shall submit one entry in response to this Challenge. Team entries must have an individual identified as the primary point of contact and prize recipient. By submitting an entry, a Participant authorizes their institution’s name to be released to the media.

The submission package must include:

* white paper
* corresponding tool

#### Judging Criteria

**Judging Panel**

Each team will initially be provided with a dataset consisting of recorded camera imagery of vessel traffic along with the recorded GPS track of a vessel of interest that is seen in the imagery. Submitted solutions will be evaluated against additional camera data correlated to recorded vessel tracks that are not included in the competition testing set in order to verify generalization of the solutions. The same vessel and the same instrumentation will be used in both the competition data set and the judging data set.

Submitted software entries will be executed on a Nvidia Jetson TX2 running the Ubuntu 18.04 operating system and preconfigured with Robot Operating System (ROS) software. This system will have no access to the internet and will be set up solely for this challenge.

Specific questions about software availability, version, or compatibility concerns should be directed to the challenge managers during the course of the challenge and should be fully resolved before final submission of an entry.

**Judging Criteria**

Track Accuracy

This will calculate the Root Mean Square Error (RMSE) between the ground truth tracks and the submitter’s
It will have a weight of 70% towards the final score

Overall Processing Time

This is the time from when your software starts until your software ends and a list of waypoints is generated
It will have a weight of 30% towards the final score

## Challenge Overview

To aim of this challenge is to develop a computer vision system that will plot the tracks of shipping traffic exclusively using the **passive** sensing capability of onboard cameras.

## [Dataset](https://drive.google.com/drive/folders/1Eq7afvav49OmWo5iNSKL7fEJk-TGNp0V?usp=sharing) README

There are 3 independent datasets:

1) 2 GPS data on the target boat (main boat and mobile device on boat)
2) GPS data on the video boat
3) Video footage on the video boat


