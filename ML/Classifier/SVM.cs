using System;
using System.Linq;
using Accord.Statistics.Models.Regression.Linear;
using Accord.Statistics.Analysis;
using Accord.IO;
using Accord.Math;
using System.Data;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Kernels;
using Accord.Controls;
using Accord.MachineLearning;
using Accord.MachineLearning.VectorMachines;

namespace ML.Classifier
{
    /// <summary>
    /// Метод опорных векторов
    /// </summary>
    class SVM : IMethodLearning<MulticlassSupportVectorMachine<Linear>>
    {
        /// <summary>
        /// Обучающаяся выборка - входные параметры
        /// </summary>
        public double[][] DataTrainInput { get; }

        /// <summary>
        /// Обучающаяся выборка - входные параметры
        /// </summary>
        public int[] DataTrainOutput { get; }

        public SVM(double[][] dataTrainInput, int[] dataTrainOutput)
        {
            DataTrainInput = dataTrainInput;
            DataTrainOutput = dataTrainOutput;
        }

        public MulticlassSupportVectorMachine<Linear> MachineLearning()
        {
            // Create a one-vs-one multi-class SVM learning algorithm 
            var teacher = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            return teacher.Learn(DataTrainInput, DataTrainOutput);
        }
    }
}


//        DataTable data_train = new CsvReader(@"H:\Documents\Visual Studio 2015\Projects\ML\ML\train.csv", true).ToTable();
//            DataTable data_test = new CsvReader(@"H:\Documents\Visual Studio 2015\Projects\ML\ML\train_lite.csv", true).ToTable();

//            // Convert the DataTable to input and output vectors (train and test)
//            int[] trainOutputs = data_train.Columns["label"].ToArray<int>();
//            data_train.Columns.Remove("label");
//            double[][] trainInputs = data_train.ToJagged<double>();

//            int[] testOutputs = data_test.Columns["label"].ToArray<int>();
//            data_test.Columns.Remove("label");
//            double[][] testInputs = data_test.ToJagged<double>();

//            //// Now, we can create the sequential minimal optimization teacher
//            //var learn = new SequentialMinimalOptimization()
//            //{
//            //    UseComplexityHeuristic = true,
//            //    UseKernelEstimation = false
//            //};

//            // Create a one-vs-one multi-class SVM learning algorithm 
//            var teacher = new MulticlassSupportVectorLearning<Linear>()
//            {
//                // using LIBLINEAR's L2-loss SVC dual for each SVM
//                Learner = (p) => new LinearDualCoordinateDescent()
//                {
//                    Loss = Loss.L2
//                }
//            };

//            var machine = teacher.Learn(trainInputs, trainOutputs);

//            int[] predicted = machine.Decide(testInputs);

//            // print result
//            int i = 0;
//            Console.WriteLine("results - (predict ,real labels)");
//            foreach (int pred in predicted)
//            {

//                Console.Write("({0},{1} )", pred, testOutputs[i]);
//                i++;
//            }

//            //calculate the accuracy - Compute classification error
//            double error = new ZeroOneLoss(testOutputs).Loss(predicted);

//            Console.WriteLine("\n accuracy: {0}", 1 - error);

//            Console.ReadLine();
//        }
//    }
//}
