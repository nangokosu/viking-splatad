# Radar-Enhanced 3D Tree Trunk Reconstruction in Foliage-Occluded Forest Environments

Text transcription of `Final_Forest_Reconstruction_Proposal.pdf` (same folder; the PDF is authoritative).
Kept so code and docs can cite sections and equations by number. Editorial notes are marked **[note]**.

**Team:** Dorsa Molaverdikhani · Tuan Khai Nguyen · Neel Joshi

## 1. Motivation
The Viking Hills dataset is the first forest mapping dataset with co-registered LiDAR, camera and radar data [1]. It was collected by a ground robot carrying a 32-beam LiDAR, a 4D imaging radar and a camera.

Seasonal vegetation growth is a key challenge. In June, tall grass blocks LiDAR measurements, and trunk segmentation IoU drops by 7–14 percentage points compared with May [1]. Radar is less affected by foliage and can detect trunks beyond 30 m. However, its lateral resolution is limited (about 0.87 m at 40 m), which makes precise trunk reconstruction difficult [1].

These complementary strengths motivate combining LiDAR, camera and radar through 3D Gaussian Splatting (3DGS) [2].

**Research question:** *Does incorporating 4D imaging radar into a LiDAR-camera 3D Gaussian Splatting pipeline improve tree trunk reconstruction in foliage-occluded forest environments?*

## 2. Related Work
- **Multimodal 3DGS targets driving and uses two modalities.** SplatAD [3] uses camera and LiDAR, RaGS [4] uses camera and radar, and RadarSplat [6] also uses radar. All were built and evaluated on road-driving data.
  - They have little evaluation in forests, where thin trunks are partly hidden by dense grass and foliage.
  - They rely on one or two sensing modalities. A third modality could fill in where individual sensors are occluded or limited in resolution.
- **3DGS forest reconstruction uses cameras and LiDAR.** One example is Zhou et al. [5], which combines handheld LiDAR with UAV imagery. Adding 4D radar to 3DGS forest reconstruction remains largely unexplored.

## 3. Method
The method builds on SplatAD [3], an open-source, real-time 3DGS framework for LiDAR and camera. Radar is added in two complementary ways.

### 3.1 Radar-Guided Gaussian Initialization
3DGS optimises an initial set of Gaussians [2]. SplatAD initialises them from camera and LiDAR data, but its architecture allows custom-initialised Gaussians to be added.

The plan is a custom data parser that processes Viking Hills radar measurements. It generates extra Gaussians for tree trunks in regions where dense foliage weakens or blocks LiDAR returns but radar still detects them. These radar-informed Gaussians supplement SplatAD's existing initialization.

### 3.2 Radar Loss Term: Vegetation-Aware Weighting
Vegetation obstruction means vegetation along the line of sight between a sensor and a candidate 3D point. The Viking Hills baseline [1] uses MinkowskiUNet [9] for radar point cloud semantic segmentation.

The baseline loss is a standard cross-entropy over radar points:

- (Eq. 1) L_i = −log p_i(y_i), where p_i(y_i) is the predicted probability of the ground-truth class y_i.
- (Eq. 2) L_base = (1/N) Σ_{i=1..N} L_i, where N is the number of radar points.

A vegetation-aware weight gives trunk errors under dense vegetation more importance:

- (Eq. 3) w_i = 1 + α · V_i · [y_i = trunk]
- (Eq. 4) L_weighted = (1/N) Σ_{i=1..N} w_i · L_i

Here V_i is the vegetation obstruction score of point i, and [y_i = trunk] is 1 when the ground-truth class is trunk and 0 otherwise.

**[note]** The PDF calls the weighting strength "λ ≥ 0" in the text but writes α in Eq. 3. This project uses **α ≥ 0**.

**Binary vegetation obstruction score:**

- (Eq. 5) V_i = 0 for May (short grass), and V_i = 1 for June (tall grass).

**Continuous line-of-sight obstruction score (stretch goal).** A coarse 3D occupancy grid is built from the LiDAR point clouds and the camera calibration:

- (Eq. 6) G(c) = 1 if vegetation is observed in cell c; 0 if c is non-vegetation or free space; "?" if unknown.

For each radar measurement i, let r_i be the ray from the sensor to the 3D point. N_v(r_i) is the number of vegetation cells the ray crosses, and N_k(r_i) is the number of crossed cells with a known class.

- (Eq. 7) V_i = N_v(r_i) / N_k(r_i), defined when N_k(r_i) > 0.

V_i lies in [0, 1]. A ray with no known cells has no defined score and must be handled separately.

**[note: project adaptation]** In this project the radar loss runs **inside SplatAD training** and is label-free:
- an opacity loss, −log σ(o), on Gaussians inside each radar seed's resolution ellipsoid;
- optionally, a one-sided radar depth term.

Radar is never rendered. See `docs/PLAN.md`.

## 4. Evaluation
- **May baseline.** Train on May data, when vegetation is sparse. Render held-out camera poses and compare with the ground-truth images using PSNR [7] and SSIM [8].
- **May + June.** Train on both months and evaluate on held-out June poses, where tall grass hides the trunks. Assess trunk visibility and completeness qualitatively, and visualise the 3D occupancy grid to find regions where occlusion hurt reconstruction.

**[note: current scope]** June only, two runs: SplatAD versus SplatAD with radar init and radar loss. May is used only as a pipeline readiness check. See `docs/PLAN.md`.

## 5. Project Timeline (six weeks)
- **Weeks 1–2:** set up SplatAD, preprocess Viking Hills, verify calibration, and run the May baseline.
- **Week 3:** build the radar data parser and radar-informed Gaussians.
- **Week 4:** build the binary vegetation-weighted radar loss. The continuous score is a stretch goal.
- **Week 5:** evaluate baseline against radar-enhanced runs (PSNR, SSIM, trunk visibility, occupancy grids).
- **Week 6:** run ablations, analyse results, and write the final report and presentation.

## References
1. M. Kubelka et al., "Viking Hill Dataset: A Lidar-Radar-Camera Dataset for Detection and Segmentation in Forest Scenes," arXiv:2606.19154, 2026.
2. B. Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering," ACM TOG 42(4), 2023.
3. G. Hess et al., "SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving," CVPR 2025. arXiv:2411.16816.
4. X. Bai et al., "RaGS: Unleashing 3D Gaussian Splatting from 4D Radar and Monocular Cue for 3D Object Detection," CVPR 2026. arXiv:2507.19856.
5. Z. Zhou et al., "LiDAR-Guided Semantic 3D Gaussian Splatting for Forest Digital Twins," Remote Sensing 18(11):1696, 2026.
6. P.-C. Kung et al., "RadarSplat: Radar Gaussian Splatting for High-Fidelity Data Synthesis and 3D Reconstruction of Autonomous Driving Scenes," ICCV 2025. arXiv:2506.01379.
7. A. Hore and D. Ziou, "Image Quality Metrics: PSNR vs. SSIM," ICPR 2010.
8. Z. Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity," IEEE TIP 13(4), 2004.
9. C. Choy, J. Gwak and S. Savarese, "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks," CVPR 2019.
