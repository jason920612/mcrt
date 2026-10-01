# MCRT

用於 Minecraft Java 26.3（Fabric）的路徑追蹤寫實渲染器。寄生在官方的 Vulkan 後端上，渲染核心用 C++ 撰寫。

## 需求

- 支援硬體光追的顯卡（開發機：RTX 4060）
- JDK 25、Visual Studio 2022 Build Tools（C++）、CMake 3.25 以上
- Slang 編譯器，放在 `tools/slang/`（從 https://github.com/shader-slang/slang/releases 下載 windows-x86_64 版並解壓到這裡）

## 建置與執行

```
./gradlew build                  # 會一併編譯 native/ 和 shaders/，DLL 打包進 jar
./gradlew runClient              # 開發用客戶端（run/options.txt 需設定 preferredGraphicsBackend:"vulkan"）
./gradlew runClient -PmcrtDebug=1      # 除錯：1 = 只顯示 albedo，2 = 顯示法線
./gradlew runClient -PmcrtDevTime=6000 # 開發用：把單人世界固定在某個時間（6000 = 正午）並設為晴天
```

執行期間，日誌每 5 秒會輸出一行 `[stats]`，包含 FPS、我們這個 pass 的 GPU 耗時、常駐區段數、待處理區段數、TLAS instance 數，以及累積的幀數。

## 架構

- `src/main/java/dev/mcrt/`：Fabric 模組
  - `VulkanFeatureSetsMixin`：把光追擴充列為 Vulkan 裝置的必要功能
  - `GameRendererMixin`：在 `renderLevel()` 之後呼叫原生渲染器
  - `rt/RtRenderer`：透過 FFM 把畫面資料交給 C++，再把錄好的 command buffer 用 `VulkanCommandEncoder.execute()` 接進 MC 自己的 submit 順序
- `native/`：C++20 核心（volk、VMA、Vulkan 1.2 + KHR 光追擴充），共用 MC 的 `VkDevice`
- `shaders/`：Slang 著色器，建置時編譯成 SPIR-V 並嵌入 DLL

MC 的 Vulkan 後端有兩個慣例要注意：所有影像都維持 `GENERAL` layout，並用全域 memory barrier 同步；主畫面緩衝是 `R8G8B8A8_UNORM`，記憶體中由下往上存放（OpenGL 風格），所以 NDC 的 y 軸要對應成 `uv.y*2-1`。
