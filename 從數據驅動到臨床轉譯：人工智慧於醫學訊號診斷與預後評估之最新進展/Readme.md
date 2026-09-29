# 從數據驅動到臨床轉譯：人工智慧於醫學訊號診斷與預後評估之最新進展
* 講者:臺北醫學大學 彭徐鈞教授
* 時間:2026/09/22



## 1. 研究動機 (Research Motivation)

1. **特發性黃斑部皺褶／黃斑上膜（Idiopathic Epiretinal Membrane, iERM）盛行率高**：
   - 在一般人群中盛行率約為 **7% ~ 11%**。
   - 患者會出現視力模糊、視物變形（Metamorphopsia）等症狀。
2. **術後視力恢復期長且成效難以預測**：
   - 玻璃體切除併膜剝除手術後，神經纖維與感光細胞受損之修復通常需 **3 至 6 個月，甚至長達 12 個月**。
   - 臨床上即使手術成功去除黃斑皺褶、解剖結構厚度顯著恢復正常，**部分患者的視力（BCVA）卻依然無法顯著改善**。
3. **現有光學相干斷層掃描（OCT）特徵缺乏定量整合標準**：
   - 雖然醫學界已提出多種 OCT 生物標記（Biomarkers，如 EIFL、DRIL、EZ 破壞等），但各標記與最終視力預後之間的量化關聯複雜，目前缺乏標準化準則直接轉換成精確的視力預測。
4. **真實臨床資料取得不易且具主觀性**：
   - 臨床患者往往無法精確說明視力下降或扭曲的起點時間，或伴隨難以釐清的潛在共病；若能**僅依賴術前客觀的 OCT 影像**進行精準預後預測，將具備高度的臨床轉譯可行性與實用價值。

---

## 2. 研究目的 (Research Objective)

1. **建立自動化深度學習預測模型**：
   - 僅使用術前常規黃斑部 OCT 影像，預測 iERM 患者在接受手術 **1 年後** 的最佳矯正視力（BCVA）改善程度。
2. **量化二元分類視覺預後**：
   - **顯著改善組（Pronounced Visual Improvement, P-IM）**：術後 1 年 BCVA 於 Snellen 視力表進步 $\ge 2$ 行。
   - **有限改善組（Limited Visual Improvement, L-IM）**：術後 1 年 BCVA 於 Snellen 視力表進步 $< 2$ 行。
3. **提供具可解釋性的臨床輔助決策依據**：
   - 透過熱圖分析（Grad-CAM）找出影響預後預測的關鍵視網膜微結構生物標記。
   - 輔助一般眼科醫師（評估是否需轉診）與視網膜專科醫師（決定手術時機與合併手術策略，如合併白內障手術或保守觀察）。

---

## 3. 解決方法與技術說明 (Methodology & Technical Description)

### 3.1 總體研究框架與流程 (Research Pipeline)
```
[術前 OCT 影像] 
       │
       ▼
[影像前處理 (Image Preprocessing)]
  ├── 影像縮放 (Resize)
  ├── 視網膜感興趣區域分割 (ROI Cropping)
  └── 資料增強 (Data Augmentation: 水平翻轉 Horizontal Flip)
       │
       ▼
[深度學習模型訓練 (Deep Learning Models)]
  ├── Inception-v3 (輸入尺寸: 299 × 299)
  ├── ResNet-101   (輸入尺寸: 224 × 224) ─── 表現最佳
  └── VGG-19       (輸入尺寸: 224 × 224)
  └── 訓練優化器: SGDM / 5-Fold Cross Validation
       │
       ▼
[二元分類預後輸出 (Binary Classification)]
  ├── P-IM (視力進步 ≥ 2 行)
  └── L-IM (視力進步 < 2 行)
       │
       ▼
[模型可解釋性 (Explainable AI)]
  └── Grad-CAM 熱圖生成 (對焦於視網膜內外層結構完整性)
```

### 3.2 資料集與收案標準 (Dataset & Clinical Cohorts)
- **資料來源**：臺北市立醫院仁愛院區（Renai branch）與忠孝院區（Zhongxiao branch），時間區間 2015/07/01 ～ 2022/07/31。
- **篩選過程**：總計篩選 1,773 名患者（1,839 眼），排除次發性 ERM、第 1 期輕微 ERM、增殖性糖尿病視網膜病變、術後復發、其他眼內手術史或影像品質不良者，最終納入 653 名患者（696 眼）。
  - **訓練／驗證集（仁愛院區）**：607 名患者（644 眼 / 644 張影像）
    - P-IM：416 例
    - L-IM：228 例
  - **外部測試集（忠孝院區）**：46 名患者（52 眼 / 52 張影像）
    - P-IM：29 例
    - L-IM：23 例
- **影像儀器**：Optovue RTVueModel-RT 100 scanner 與 Carl Zeiss Cirrus 5000-HD-OCT（以黃斑中心凹為中心之標準 10-mm 水平/垂直 B-scans）。

### 3.3 影像前處理與特徵萃取 (Preprocessing)
- **軟體環境**：MATLAB R2023b (MathWorks, Natick, MA)。
- **ROI 裁切**：精確鎖定視網膜內外層（Inner & Outer Retina）組織範圍，去除無關背景雜訊。
- **資料增強（Data Augmentation）**：針對訓練集類別不平衡進行平衡擴充（如 P-IM 擴增 2 倍至 666 例、L-IM 擴增 4 倍至 728 例）。

### 3.4 深度學習架構與優化策略
- 測試並評估三種代表性卷積神經網路架構：**ResNet-101、Inception-v3、VGG-19**。
- 優化器採用 **SGDM（Stochastic Gradient Descent with Momentum）**。
- 採用 **5 折交叉驗證（5-fold cross validation）** 確保模型泛化能力與穩定度。

### 3.5 可解釋性 AI（XAI / Grad-CAM）
- 利用 **Grad-CAM (Gradient-weighted Class Activation Mapping)** 生成注意力熱圖。
- 臨床微結構生物標記對應發現：
  - **P-IM（顯著改善組）**：熱圖關注焦點在於視網膜**內層與外層結構完整性（Inner + Outer layer integrity）**。
  - **L-IM（有限改善組）**：熱圖關注焦點呈現視網膜**內層組織結構破壞或破壞性病變（Inner layer destructiveness）**。

---

## 4. 臨床應用價值與研究限制 (Clinical Applications & Limitations)

### 臨床應用場景
1. **一般眼科醫師（General Ophthalmologists）**：
   - 輔助評估患者黃斑上膜病情是否嚴重至需要轉診至視網膜專科。
2. **視網膜專科醫師（Retinal Specialists）**：
   - **高齡／健康狀況差／順從性低患者**：若預測術後預後不佳，可建議優先保守觀察，避免承受不必要的手術風險。
   - **同時合併 ERM 與白內障患者**：
     - 若模型預測預後良好 $\rightarrow$ 建議同時進行白內障與 ERM 膜剝除合併手術。
     - 若模型預測預後不佳 $\rightarrow$ 建議單純施作白內障手術即可，降低手術負擔。

### 研究限制
1. 樣本數規模仍屬中小型（單一醫療體系兩院區）。
2. 術後視力僅切分為二元分類（進步 $\ge 2$ 行 vs $< 2$ 行），未來可朝連續性視力數值回歸或多元分級拓展。
3. OCT 儀器機型種類有限（Optovue / Zeiss）。
4. 影像感興趣區域（ROI）目前包含部分人工前處理介入。

---

## 5. 演講延伸：其他醫學 AI 研發成果統整 (Related AI Works in the Presentation)

| 研究主題 | 臨床問題與目的 | 使用技術與演算法 | 關鍵成果與應用 |
| :--- | :--- | :--- | :--- |
| **兒童幕上低級別膠質瘤癲癇預測**<br>*(Supratentorial Low-Grade Glioma in Children)* | 預測兒童膠質瘤是否併發相關癲癇症狀 | T2-FLAIR MRI 影像形態定量與放射組學（Radiomics）、特徵篩選、機器學習分類器 | 結合腫瘤解剖位置與紋理強度特徵，精準識別癲癇高風險兒童膠質瘤 |
| **重度憂鬱症（MDD）治療預後評估** | 預測抗憂鬱治療在第 4、6、8 週 HAMD 評分改善反應 | 腦電波（EEG）功率分析（AP/RP）、相位同步分析（PLV/PLI）、19 種 ML 模型與 LOOCV | 僅透過第 0~1 週早期腦波變化即可早期預測抗憂鬱劑療效反應 |
| **聽神經瘤（Vestibular Schwannoma）放療反應預測** | 聽神經瘤體積監測及放療後「偽進展 vs 真實縮小」之鑑別 | 基於 **nnU-Net** 自動化 3D 分割、放射組學特徵（小波轉換）+ **ExtraTrees** 兩階段分類預測 | 建置完整圖形化介面（GUI），整合 MRI 影像讀取、自動分割、體積計算與預後預測 |
