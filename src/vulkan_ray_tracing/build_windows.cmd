@echo off
setlocal enabledelayedexpansion

:: Exit immediately on error
set ERRLEVEL=0

:: === Set up variables ===
set BUILD_DIR=build
set EXT_PATH=external

:: Set Vulkan SDK (make sure VULKAN_SDK is set in your env)
set "VULKAN_SDK=%VULKAN_SDK%\Include"
set "VULKAN_SDK_INCLUDE=%VULKAN_SDK%\Include"
set "VULKAN_SDK_LIB=%VULKAN_SDK%\Lib"

echo Vulkan SDK Path: %VULKAN_SDK%
echo Include: %VULKAN_SDK_INCLUDE%
echo Lib: %VULKAN_SDK_LIB%

:: Clean and recreate build directory
if exist %BUILD_DIR% rmdir /s /q %BUILD_DIR%
mkdir %BUILD_DIR%
cd %BUILD_DIR%

:: === Handle external dependencies ===

:: SPIR-V Headers
set SPIRV_HEADERS_BIN=%EXT_PATH%\spirv-headers
xcopy /E /I /Y ..\%SPIRV_HEADERS_BIN% %SPIRV_HEADERS_BIN%
mkdir %SPIRV_HEADERS_BIN%\build
cd %SPIRV_HEADERS_BIN%\build
cmake ..
cmake --build . --config Release
cd ..\..\..

:: SPIR-V Tools
set SPIRV_BIN=%EXT_PATH%\spirv-tools
xcopy /E /I /Y ..\%SPIRV_BIN% %SPIRV_BIN%
mkdir %SPIRV_BIN%\build
cd %SPIRV_BIN%\build
cmake -DSPIRV-Headers_SOURCE_DIR=%CD%\..\..\spirv-headers ..
cmake --build . --config Release
cd ..\..\..

:: GLSLang
set GLSLANG_BIN=%EXT_PATH%\glslang
xcopy /E /I /Y ..\%GLSLANG_BIN% %GLSLANG_BIN%
mkdir %GLSLANG_BIN%\build
cd %GLSLANG_BIN%\build
cmake ^
  -DENABLE_OPT=OFF ^
  -DSPIRV_TOOLS_INCLUDE_DIR=%CD%\..\..\spirv-tools\include ^
  -DSPIRV_TOOLS_LIB_DIR=%CD%\..\..\spirv-tools\build ^
  ..
cmake --build . --config Release
cd ..\..\..

:: VCPKG
set VCPKG_BIN=%EXT_PATH%\vcpkg
xcopy /E /I /Y ..\%VCPKG_BIN% %VCPKG_BIN%
cd %VCPKG_BIN%
call bootstrap-vcpkg.bat
vcpkg integrate install
vcpkg install tbb
cd ..\..

:: pybind11
set PYBIND11_BIN=%EXT_PATH%\pybind11
xcopy /E /I /Y ..\%PYBIND11_BIN% %PYBIND11_BIN%

:: GLM
set GLM_BIN=%EXT_PATH%\glm
xcopy /E /I /Y ..\%GLM_BIN% %GLM_BIN%

:: === Configure with CMake ===
echo Building main Vulkan project...
for /f "delims=" %%i in ('where python') do (
    if not defined PYTHON_EXECUTABLE set PYTHON_EXECUTABLE=%%i
)
echo python is %PYTHON_EXECUTABLE%

cmake ^
  -G "NMake Makefiles" ^
  -DENABLE_OPT=OFF ^
  -DGLM_INCLUDE_DIR=%CD%\%EXT_PATH%\glm ^
  -DCMAKE_PREFIX_PATH=%CD%\%EXT_PATH%\vcpkg\installed\x64-windows ^
  -DBASE_DIRECTORY=%CD%\%EXT_PATH% ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DVulkan_INCLUDE_DIRS=%VULKAN_SDK_INCLUDE% ^
  -DVulkan_LIBRARIES=%VULKAN_SDK_LIB% ^
  -DPython3_EXECUTABLE=%PYTHON_EXECUTABLE% ^
  ..

:: === Build the project ===
nmake install

echo Build complete. You can now run your Vulkan app.
