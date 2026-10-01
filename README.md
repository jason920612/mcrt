# MCRT

用於 Minecraft Java 26.3（Fabric）的路徑追蹤寫實渲染器。寄生在官方的 Vulkan 後端上，渲染核心用 C++ 撰寫。

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
```

執行期間，日誌每 5 秒會輸出一行 `[stats]`，包含 FPS、GPU 耗時（光追、降噪、合成的總和）、常駐區段數、待處理區段數、TLAS instance 數，以及發光方塊數。

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
  - `pathtrace.slang`：路徑追蹤（太陽或月亮直射光、發光方塊 RIS 取樣、一次漫反射反彈），輸出 G-buffer
  - `denoise.slang`：時間重投影累積、à-trous 空間濾波、自動曝光、ACES tonemap

## 材質

`tools/materials/materials.json` 定義兩件事：每種材質的來源（全部是 ambientCG 的 CC0 素材），以及哪些方塊的哪個面（頂面、側面、底面）使用哪種材質。`./gradlew build` 會執行 `build_materials.py`，下載素材並打包成兩張圖：

- 顏色圖：RGB 是顏色，A 是高度
- 資料圖：法線 xy、粗糙度、AO

結果輸出到 `build/generated/materials`，不納入 git。素材出處會寫進同目錄的 `CREDITS.md`。要新增材質，就在 json 裡加一筆來源和方塊對應。`scale` 是一張貼圖覆蓋幾格方塊；設了 `tinted` 的材質會依生態域顏色調色。

MC 的 Vulkan 後端有兩個慣例要注意：所有影像都維持 `GENERAL` layout，並用全域 memory barrier 同步；主畫面緩衝是 `R8G8B8A8_UNORM`，記憶體中由下往上存放（OpenGL 風格），所以 NDC 的 y 軸要對應成 `uv.y*2-1`。
