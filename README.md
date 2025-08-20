# 🎵 Music File Sorter
一个基于多线程的音乐文件分类整理工具，可解析文件名中的歌曲信息，根据 Excel 表匹配歌手并分类到对应文件夹，支持日志记录和去重处理。

## ✨ Features
- 📂 对“.mp3”、“.jpg”、“.lrc”格式文件进行递归文件夹扫描
- ⚡ 采用多线程处理以实现高性能
- 🔍 解析文件名的元数据（歌曲名、艺术家、专辑、时长、文件类型）
- 📊 将文件分类到以下类别：
  - 欧美男歌手
  - 欧美女歌手
  - 欧美乐队
  - 其他
- 📝 将进度信息记录到控制台并轮转日志文件
- 📑 将不匹配的名称和缺失的元数据导出到 Excel 文件（`non_name.xlsx`、`non_artist.xlsx`）
- 🔒 对 Windows 路径进行安全清理并处理重复的文件名

## File Structure
C:\music\sorted\
 ├── 欧美男歌手\
 │    ├── Ed Sheeran\
 │    │     ├── Shape of You.mp3
 │    │     └── Shape of You.jpg
 ├── 欧美女歌手\
 │    ├── Adele\
 │    │     ├── Hello.mp3
 │    │     └── Hello.lrc
 ├── 欧美组合\
 │    ├── Coldplay\
 │    │     └── Yellow.mp3
 └── 其他\
      └── Unknown.mp3



# 🎵 Music File Sorter

A multithreaded tool for classifying and organizing music files (`.mp3`, `.jpg`, `.lrc`) by parsing metadata from filenames.  
Supports Excel-based artist name matching, logging, duplicate handling, and outputting missing data to Excel.

---

## ✨ Features
- 📂 Recursive folder scanning for `.mp3`, `.jpg`, `.lrc`
- ⚡ Multithreaded processing for high performance
- 🔍 Parse filename metadata (song name, artist, album, duration, file type)
- 📊 Classify files into categories:
  - Western Male Singer
  - Western Female Singer
  - Western Group
  - Others
- 📝 Log progress to console & rotating log files
- 📑 Export unmatched names and missing metadata to Excel (`non_name.xlsx`, `non_artist.xlsx`)
- 🔒 Safe Windows path sanitization & duplicate filename handling

---


