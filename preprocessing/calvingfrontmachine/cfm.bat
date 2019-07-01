PATH D:\ProgramData\Anaconda3\Scripts;D:\ProgramData\Anaconda3;%PATH%
SETLOCAL
SET PYTHONHOME=
C:
cd C:\Users\Daniel\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\calvingfrontmachine
call D:\ProgramData\Anaconda3\Scripts\activate.bat D:\ProgramData\Anaconda3\envs\cfm
python calving_front_machine_cnn.py %1 %2 %3 %4
REM exit