# MCRT

用於 Minecraft Java 26.3（Fabric）的路徑追蹤寫實渲染器。寄生在官方的 Vulkan 後端上，渲染核心用 C++ 撰寫。

設計原則：遊戲本身（碰撞、放置與破壞方塊、生物、紅石）完全維持原版，方塊只被當成「這裡是什麼材質」的標記；畫面上的地形、植被和遠景由 MCRT 自己重新生成，目標是風景畫等級的寫實畫面。

## 需求

- 支援硬體光追的顯卡（開發機：RTX 4060）
- JDK 25、Visual Studio 2022 Build Tools（C++）、CMake 3.25 以上
- Slang 編譯器，放在 `tools/slang/`（從 https://github.com/shader-slang/slang/releases 下載 windows-x86_64 版並解壓到這裡）
- Python 3 和 Pillow（材質工具會用到；第一次建置時會從 ambientCG 下載 CC0 材質）

## 建置與執行

```
./gradlew build                  # 會一併編譯 native/ 和 shaders/，DLL 打包進 jar
./gradlew runClient              # 開發用客戶端（run/options.txt 需設定 preferredGraphicsBackend:"vulkan"）
./gradlew runClient -PmcrtDebug=1      # 除錯：1 = 只顯示 albedo，2 = 顯示法線
./gradlew runClient -PmcrtDevTime=6000 # 開發用：把單人世界固定在某個時間（6000 = 正午）並設為晴天
./gradlew runClient -PmcrtDevSpin=1.5  # 開發用：玩家每 tick 自動旋轉的角度，用來測試移動中的降噪
./gradlew runClient "-PmcrtDevCommands=setblock ~ ~ ~3 glowstone;..."  # 開發用：進入世界後執行一次的指令（用分號分隔）
./gradlew runClient -PmcrtCheckerboard=false  # 關閉棋盤格光照（預設開啟：每幀只有一半像素計算光照，由降噪器補齊）
./gradlew runClient -PmcrtDisable             # 停用光追，用原版 Vulkan 渲染做效能對照（日誌會輸出 [vanilla] fps）
./gradlew runClient -PmcrtRenderScale=0.67    # 以 67% 解析度渲染，再由 TAAU 放大回視窗解析度（0.5～1）
./gradlew runClient -PmcrtFarTerrain=false    # 關閉渲染距離外的遠景
```

執行期間，日誌每 5 秒會輸出一行 `[stats]`，包含 FPS、GPU 耗時（起點時間戳會包含等待前面工作的時間，偏高）、原生端每幀的 CPU 耗時、光源清單重建耗時、常駐區段數、待處理區段數、TLAS instance 數，以及發光方塊數。

## 架構

- `src/main/java/dev/mcrt/`：Fabric 模組
  - `VulkanFeatureSetsMixin`：把光追擴充列為 Vulkan 裝置的必要功能
  - `LevelRendererMixin`：在 MC 的 frame graph 裡插入光追步驟，取代原版的地形和天空繪製（實體、粒子等仍由原版畫）
  - `SectionCompileTaskMixin`、`RenderSectionMixin`：擷取 MC 編譯好的區段網格，並追蹤區段的載入與卸載
  - `LevelExtractorMixin`：補編譯視錐外的區段，讓背後的地形也能投影和反彈光線
  - `GameRendererMixin`：擷取含視角晃動的投影矩陣，以及關閉時釋放資源
  - `rt/RtRenderer`：透過 FFM 把畫面資料交給 C++，再把錄好的 command buffer 用 `VulkanCommandEncoder.execute()` 接進 MC 自己的 submit 順序
  - `rt/SectionScanner`：掃描區段內的發光方塊（依方塊種類指定光色）和材質對應
  - `rt/MaterialRegistry`：讀取材質表，用 MC 的 NativeImage 解碼材質圖並交給 C++
  - `SectionRegionLightingMixin`：光追啟用時，取消 MC 預先烘進頂點顏色的面向明暗
- `native/`：C++20 核心（volk、VMA、Vulkan 1.2 + KHR 光追擴充），共用 MC 的 `VkDevice`
- `shaders/`：Slang 著色器，建置時編譯成 SPIR-V 並嵌入 DLL
  - `pathtrace.slang`：路徑追蹤（太陽或月亮直射光、發光方塊 RIS 取樣加上時間性 ReSTIR 重用、一次漫反射反彈），輸出 G-buffer
  - `denoise.slang`：時間重投影累積、à-trous 空間濾波、空氣透視（大氣霧）、自動曝光
  - `taa.slang`：TAAU（時間性反鋸齒＋放大）、ACES tonemap，並把深度放大給 MC 的實體繪製使用

## 材質

`tools/materials/materials.json` 定義兩件事：每種材質的來源（全部是 ambientCG 的 CC0 素材），以及哪些方塊的哪個面（頂面、側面、底面）使用哪種材質。`./gradlew build` 會執行 `build_materials.py`，下載素材並打包成兩張圖：

- 顏色＋高度（BC3）
- 法線 xy（BC5）、粗糙度＋AO（BC5）

植被貼圖（草葉、闊葉、針葉）不是下載的，而是由 `tools/materials/procedural.py` 程序生成，沒有授權問題。`card` 標記這類以 alpha 測試繪製的植被卡片材質；`relief` 決定該材質的地形起伏方式；`gain` 可調亮或調色（例如冰的照片比遠看的冰暗）。

每種材質輸出成一個 `.mcm` 檔：預先算好的 mipmap，並以 BC3（顏色＋高度）與兩張 BC5（法線、粗糙度＋AO）壓縮，編碼器是 `bc_encode.py`。輸出到 `build/generated/materials`，不納入 git。素材出處會寫進同目錄的 `CREDITS.md`。要新增材質，就在 json 裡加一筆來源和方塊對應。`scale` 是一張貼圖覆蓋幾格方塊；設了 `tinted` 的材質會依生態域顏色調色。

## 地形（方塊只是材質標記）

在 `materials.json` 裡標記 `"smooth": true` 的自然方塊（石頭、泥土、草、沙、礫石、雪、冰等）不再以方塊繪製，而是重新生成成連續的地形，只影響外觀，碰撞箱仍然是方塊（`native/src/terrain_mesher.cpp`）。

- **密度場**：每半格取樣一次，值是周圍方塊的平滑加權投票（實心 +1、空氣 −1）。
- **依材質的起伏**：岩石有稜線和水平岩層、泥土有土塊起伏、沙有沙丘、雪有積雪起伏。
- **維持遊戲形狀**：每個方塊中心都固定在正確的一側，所以單一方塊和一格寬的洞都不會消失，畫面與碰撞箱的差距大約在半格以內。碰到人造方塊時，該面會被完整覆蓋，避免看穿被剔除的面。
- **網格與材質**：用 Surface Nets 產生網格。每個頂點用加了噪聲偏移的位置查詢最近方塊的材質，著色器再於三角形內依噪聲混合，所以材質邊界是自然的曲線，不會沿著方塊格線。
- **巨觀細節**：世界座標的多尺度噪聲改變反照率和法線（乾草與茂草的區塊、雪窪、岩層），遠處也不會變成單一色塊。
- **雪層**：1～3 層的雪併入下方表面，改用雪材質；雪在陡坡上也會覆蓋。
- 鄰接區段使用相同的資料（Java 端提供三格寬的邊界、材質和草色），所以接縫處完全一致。

## 植被

- **草**：草頂面上每格散布 3～8 張草葉卡片，密度依噪聲成片變化；卡片根部精確落在地形表面上。原版的像素風短草、高草和蕨類會被移除。草葉只投下部分、隨機的陰影，草地明暗才會柔和。
- **樹**：樹葉換成程序生成的闊葉／針葉（雲杉）叢卡片，依生態域的樹葉顏色輕微調色。樹冠邊角的卡片較小並往內收，被包在樹冠內部的葉子不產生卡片，所以樹的輪廓不再是方塊。原木是八角形的樹幹。

## 遠景（到地平線）

單人遊戲時，背景執行緒直接用世界生成器取樣周圍 ±2048 格的地形高度和生態域（每 16 格一點，4 條執行緒，約 6 秒），交給 C++ 建成一張高度場網格（`FarTerrain.java`、`SectionManager::enqueueFarTerrain`）。它只會出現在已載入區塊之外，海面也用水的著色（反射、深水散射），最後隨距離融入大氣霧中。多人遊戲拿不到生成器，就退回成在渲染距離邊緣融入天空。

## 天空、雲、水

- **大氣**（`shaders/atmosphere.slang`、`sky.slang`）：Rayleigh 與 Mie 散射（單次散射加上天光的二次散射，地平線才會是淡藍白色而不是灰綠色），每幀用 compute 算出 192×108 的天空查找表（sky-view LUT）。
- **空氣透視**：依照第一個可見表面（包含水面）的距離和高度加上大氣霧，霧的顏色取自該方向的地平線天空，所以遠山會偏藍、朝太陽的方向會偏亮。遠處的雲也會沉入地平線的霧中。太陽的顏色由大氣透射率決定，所以日落時會偏橘紅。夜晚有月亮、月光和程序產生的星空。
- **體積雲**（`shaders/clouds.slang`）：在 MC 的雲層高度用程序噪聲做 raymarching，光照用 Henyey-Greenstein 相位函數，會隨風飄動。每幀先算進 512×256 的查找表，主視線和水面反射都直接查表；雲也會在地面投下陰影。MC 原本的扁平雲層已停用。
- **水**：方塊材質 ID 255 保留給水面。水面有動態波浪法線、Fresnel 反射（真的發射反射光線）、太陽高光閃爍、折射，以及依水中路徑長度計算的吸收（Beer-Lambert，紅光衰減最快）。

## 其他維度與水下

- **地獄與終界**：沒有太陽、雲和大氣，光線打出世界時的亮度改用該維度的霧色。
- **攝影機在水中**：光線一開始就算在水裡，路徑上會吸收和散射；從水下看水面時有折射和全反射。攝影機在岩漿中時，畫面直接填滿岩漿色。
- **大量發光方塊（例如岩漿湖）**：只收錄外露的發光方塊；每個區段的光源清單最多 256 個，超過時分層抽樣，並給每個樣本「代表權重」以維持期望亮度。

## 實體（生物、玩家、物品）

實體仍由 MC 自己繪製（保留原本的造型和材質），但融入光追場景：

- `StagedDrawMixin` 在 MC 準備實體網格時複製一份頂點（只取 `LevelRenderer` 的實體，不含 GUI），C++ 每幀用它重建一個 BLAS（`entity_layer.cpp`）。這份幾何對主光線不可見，但陰影、水面反射與反彈光線都看得到，所以生物會在地上投下真正的影子，也會出現在倒影裡。
- `EntityRendererMixin`：MC 只依所站位置的亮度等級照亮實體；光追啟用時，若實體到太陽（或月亮）的方向被擋住，就降低它的天空光等級，讓站在樹蔭下的生物和周圍地面一樣暗。

## 光照與畫質

- **ReSTIR**：主要表面的發光方塊取樣會存成每像素的 reservoir，下一幀重投影後與新的候選合併（歷史上限 20 倍），只對最後選中的光源發射陰影光線；光源清單重建時（例如破壞火把）會作廢舊的 reservoir。
- **TAAU**：主光線每幀做 Halton 子像素抖動，`taa.slang` 用 YCoCg 方差裁切的時間累積做反鋸齒，並可從較低的內部解析度放大。

## 高度細節

- 視差遮蔽映射（POM）：沿視線在高度圖上步進，24 格內的磚、石、木有真實的凹凸深度和遮擋，距離越遠效果越淡。
- 平滑地形的頂面與側面材質依高度混合，例如草會從石縫間長出，而不是整片均勻淡出成泥土。

MC 的 Vulkan 後端有兩個慣例要注意：所有影像都維持 `GENERAL` layout，並用全域 memory barrier 同步；主畫面緩衝是 `R8G8B8A8_UNORM`，記憶體中由下往上存放（OpenGL 風格），所以 NDC 的 y 軸要對應成 `uv.y*2-1`。
