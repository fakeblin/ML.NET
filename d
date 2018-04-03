[33mcommit 10c1c16a3cda9205d9cdfe34c6b9d6bbb24e1b30[m[33m ([m[1;36mHEAD -> [m[1;32mmaster[m[33m)[m
Author: Kopyl <fakeblin@gmail.com>
Date:   Tue Apr 3 20:12:56 2018 +0500

    Hidden dir "models"

 .gitignore | 2 [32m++[m
 1 file changed, 2 insertions(+)

[33mcommit 9ae18dd7d5960a634dd6792676664871b195ba4e[m
Author: Kopyl <fakeblin@gmail.com>
Date:   Tue Apr 3 19:58:16 2018 +0500

    Add Naive Bayes, Start to do Class(kNN)

 IMethodLearning.cs |  76 [32m++++++++++++++++++++++++++++[m
 ML.csproj          |   1 [32m+[m
 NaiveBayes.cs      | 112 [32m+++++++++++++++++++++++++++++++++++++++++[m
 Program.cs         |  98 [32m+++++++++++++++++++++++++++[m[31m---------[m
 SVM.cs             |  12 [32m++[m[31m---[m
 bin/Debug/ML.exe   | Bin [31m8704[m -> [32m10752[m bytes
 bin/Debug/ML.pdb   | Bin [31m17920[m -> [32m22016[m bytes
 kNN.cs             | 145 [32m++++++++++++++++++++++++++++++++++++++++[m[31m-------------[m
 knn.bin            | Bin [31m0[m -> [32m265399789[m bytes
 obj/Debug/ML.exe   | Bin [31m8704[m -> [32m10752[m bytes
 obj/Debug/ML.pdb   | Bin [31m17920[m -> [32m22016[m bytes
 11 files changed, 377 insertions(+), 67 deletions(-)

[33mcommit 3ff71b5deff9e0486e5c4458ced79dfdd3052724[m[33m ([m[1;31morigin/master[m[33m)[m
Author: Kopyl <fakeblin@gmail.com>
Date:   Sun Apr 1 23:03:02 2018 +0500

    "Add Logit-Regression"

 ML.csproj                                          |   3 [32m+[m
 Program.cs                                         |  67 [32m+++++++++[m[31m------------[m
 SVM.cs                                             |  13 [32m+++[m[31m-[m
 bin/Debug/ML.exe                                   | Bin [31m7680[m -> [32m8704[m bytes
 bin/Debug/ML.pdb                                   | Bin [31m15872[m -> [32m17920[m bytes
 kNN.cs                                             |  28 [32m+++++++++[m
 .../DesignTimeResolveAssemblyReferencesInput.cache | Bin [31m8089[m -> [32m8222[m bytes
 obj/Debug/ML.exe                                   | Bin [31m7680[m -> [32m8704[m bytes
 obj/Debug/ML.pdb                                   | Bin [31m15872[m -> [32m17920[m bytes
 9 files changed, 71 insertions(+), 40 deletions(-)

[33mcommit f47d427ada505594f376abb13cef5a576fcf59a9[m
Author: Kopyl <fakeblin@gmail.com>
Date:   Sun Apr 1 19:07:06 2018 +0500

    Add SVM

 ML.csproj        |   1 [32m+[m
 Program.cs       |  39 [32m+++++++++++++++++[m[31m----------------[m
 SVM.cs           |  64 [32m+++++++++++++++++++++++++++++++++++++++++++++++++++++++[m
 bin/Debug/ML.exe | Bin [31m6656[m -> [32m7680[m bytes
 bin/Debug/ML.pdb | Bin [31m13824[m -> [32m15872[m bytes
 kNN.cs           |  49 [32m+++++++++++++++++++++++++++++++++++++++[m[31m---[m
 obj/Debug/ML.exe | Bin [31m6656[m -> [32m7680[m bytes
 obj/Debug/ML.pdb | Bin [31m13824[m -> [32m15872[m bytes
 8 files changed, 132 insertions(+), 21 deletions(-)

[33mcommit a88e7798d48d4835313dbd73706048e872648b9c[m
Author: Kopyl <fakeblin@gmail.com>
Date:   Sun Apr 1 18:31:55 2018 +0500

    kNN

 App.config                                         |      6 [32m+[m
 License-LGPL.txt                                   |    506 [32m+[m
 ML.csproj                                          |    111 [32m+[m
 ML.csproj.user                                     |      6 [32m+[m
 Program.cs                                         |     69 [32m+[m
 Properties/AssemblyInfo.cs                         |     36 [32m+[m
 bin/Debug/Accord.Controls.dll                      |    Bin [31m0[m -> [32m94208[m bytes
 bin/Debug/Accord.Controls.xml                      |   3081 [32m+[m
 bin/Debug/Accord.IO.dll                            |    Bin [31m0[m -> [32m69632[m bytes
 bin/Debug/Accord.IO.xml                            |   2835 [32m+[m
 bin/Debug/Accord.MachineLearning.dll               |    Bin [31m0[m -> [32m442368[m bytes
 bin/Debug/Accord.MachineLearning.xml               |  22069 [32m++++[m
 bin/Debug/Accord.Math.Core.dll                     |    Bin [31m0[m -> [32m1441792[m bytes
 bin/Debug/Accord.Math.Core.xml                     |  88500 [32m+++++++++++++[m
 bin/Debug/Accord.Math.dll                          |    Bin [31m0[m -> [32m2220032[m bytes
 bin/Debug/Accord.Math.xml                          | 118445 [32m++++++++++++++++++[m
 bin/Debug/Accord.Statistics.dll                    |    Bin [31m0[m -> [32m897024[m bytes
 bin/Debug/Accord.Statistics.xml                    |  73263 [32m+++++++++++[m
 bin/Debug/Accord.dll                               |    Bin [31m0[m -> [32m131072[m bytes
 bin/Debug/Accord.dll.config                        |      7 [32m+[m
 bin/Debug/Accord.xml                               |   8653 [32m++[m
 bin/Debug/ICSharpCode.SharpZipLib.dll              |    Bin [31m0[m -> [32m200704[m bytes
 bin/Debug/ML.exe                                   |    Bin [31m0[m -> [32m6656[m bytes
 bin/Debug/ML.exe.config                            |      6 [32m+[m
 bin/Debug/ML.pdb                                   |    Bin [31m0[m -> [32m13824[m bytes
 bin/Debug/ML.vshost.exe                            |    Bin [31m0[m -> [32m22696[m bytes
 bin/Debug/ML.vshost.exe.config                     |      6 [32m+[m
 bin/Debug/ML.vshost.exe.manifest                   |     11 [32m+[m
 bin/Debug/ZedGraph.dll                             |    Bin [31m0[m -> [32m295424[m bytes
 bin/Debug/ZedGraph.xml                             |  25730 [32m++++[m
 bin/Debug/de/ZedGraph.resources.dll                |    Bin [31m0[m -> [32m4608[m bytes
 bin/Debug/es/ZedGraph.resources.dll                |    Bin [31m0[m -> [32m4608[m bytes
 bin/Debug/fr/ZedGraph.resources.dll                |    Bin [31m0[m -> [32m4608[m bytes
 bin/Debug/hu/ZedGraph.resources.dll                |    Bin [31m0[m -> [32m4608[m bytes
 bin/Debug/it/ZedGraph.resources.dll                |    Bin [31m0[m -> [32m4608[m bytes
 bin/Debug/ja/ZedGraph.resources.dll                |    Bin [31m0[m -> [32m4608[m bytes
 bin/Debug/pt/ZedGraph.resources.dll                |    Bin [31m0[m -> [32m4608[m bytes
 bin/Debug/ru/ZedGraph.resources.dll                |    Bin [31m0[m -> [32m4608[m bytes
 bin/Debug/sk/ZedGraph.resources.dll                |    Bin [31m0[m -> [32m4608[m bytes
 bin/Debug/sv/ZedGraph.resources.dll                |    Bin [31m0[m -> [32m4096[m bytes
 bin/Debug/tr/ZedGraph.resources.dll                |    Bin [31m0[m -> [32m4608[m bytes
 bin/Debug/zh-cn/ZedGraph.resources.dll             |    Bin [31m0[m -> [32m4096[m bytes
 bin/Debug/zh-tw/ZedGraph.resources.dll             |    Bin [31m0[m -> [32m4608[m bytes
 kNN.cs                                             |     12 [32m+[m
 .../DesignTimeResolveAssemblyReferencesInput.cache |    Bin [31m0[m -> [32m8089[m bytes
 obj/Debug/ML.csproj.FileListAbsolute.txt           |     37 [32m+[m
 obj/Debug/ML.csprojResolveAssemblyReference.cache  |    Bin [31m0[m -> [32m60199[m bytes
 obj/Debug/ML.exe                                   |    Bin [31m0[m -> [32m6656[m bytes
 obj/Debug/ML.pdb                                   |    Bin [31m0[m -> [32m13824[m bytes
 ...tedFile_036C0B5B-1481-4323-8D20-8F5ADCB23D92.cs |      0
 ...tedFile_5937a670-0e60-4077-877b-f7221da3dda1.cs |      0
 ...tedFile_E7A71F73-0F8D-4B9B-B56E-8E70B10BC5D3.cs |      0
 packages.config                                    |     11 [32m+[m
 test.csv                                           |  28001 [32m+++++[m
 test_lite.csv                                      |     18 [32m+[m
 train.csv                                          |  42001 [32m+++++++[m
 train_lite.csv                                     |     19 [32m+[m
 57 files changed, 413439 insertions(+)
