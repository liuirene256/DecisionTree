using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text;

namespace DecisionTree
{
    internal static class Program
    {
        private static readonly string path = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.Parent.Parent.FullName + '\\';
        static void Main()
        {
            //LevelI();
            //LevelIIAndBeyond();
            LevelIII();
        }
        
        static void LevelI()
        {
            // Load training samples
            var signal = DataSet.ReadDataSet(path + "signal.dat");
            var background = DataSet.ReadDataSet(path + "background.dat");

            // Load data sample
            var data = DataSet.ReadDataSet(path + "decisionTreeData.dat");

            int bestVariableIndex = -1;
            double bestSplitValue = 0;
            double bestPurity = 1;

            // TODO: Insert code here that calculates the proper values of bestVariableIndex and bestSplitValue

            double right = 0;
            double left = 0;


            for (int var = 0; var < signal.Points[0].Variables.Length; var++)
            {
                int i = 0;

                double sLeft = 0;
                double sRight = 0;
                double bLeft = 0;
                double bRight = 0;

                double[] splits = new double[signal.Points.Count + background.Points.Count];
                double[] label = new double[signal.Points.Count + background.Points.Count];
                foreach (DataPoint point in signal.Points) //loops through points
                {
                    splits[i] = point.Variables[var];
                    label[i] = 1;
                    sRight++;
                    i++;
                }
                foreach (DataPoint point in background.Points) //loops through points
                {
                    splits[i] = point.Variables[var];
                    label[i] = 0;
                    bRight++;
                    i++;
                }
                Array.Sort(splits, label);

                double purityLeft = 0;
                double purityRight = 0;
                for (int ind = 0; ind < splits.GetLength(0) - 1; ind++)
                {
                    if (label[ind] == 0)
                    {
                        bRight--;
                        bLeft++;
                    }
                    else
                    {
                        sRight--;
                        sLeft++;
                    }
                    left = sLeft + bLeft;
                    right = sRight + bRight;

                    purityLeft = sLeft / left;
                    purityRight = sRight / right;

                    if (purityLeft > 0.5)
                    {
                        purityLeft = 1 - purityLeft;
                    }
                    else if (purityRight > 0.5)
                    {
                        purityRight = 1 - purityRight;
                    }
                    else
                    {
                        purityLeft = 2;
                    }
                    if (purityLeft + purityRight < bestPurity && left >= 50 && right >= 50)
                    {
                        bestVariableIndex = var;
                        bestPurity = purityLeft + purityRight;
                        bestSplitValue = (splits[ind] + splits[ind + 1]) / 2;
                    }
                }
            }
            Console.WriteLine(bestVariableIndex + " " + bestSplitValue);
            using var file = File.CreateText(path + "decisionTreeResultsLevelI.txt");
            file.WriteLine("Event\tPurity");

            for (int i = 0; i < data.Points.Count; ++i)
            {
                // Note that you may have to change the order of the 1 and 0 here, depending on which one matches signal.
                // 1 means signal and 0 means background
                double output = data.Points[i].Variables[bestVariableIndex] > bestSplitValue ? 1 : 0;
                file.WriteLine(i + "\t" + output);
            }
        }

        static void LevelIIAndBeyond()
        {
            // Load training samples
            var signal = DataSet.ReadDataSet(path + "signal.dat");
            var background = DataSet.ReadDataSet(path + "background.dat");

            // Load data sample
            var data = DataSet.ReadDataSet(path + "decisionTreeData.dat");

            var tree = new Tree();

            // Train the tree
            tree.Train(signal, background);

            // Calculate output value for each event and write to file
            tree.MakeTextFile(path + "decisionTreeResults.txt", data);
        }
        static void LevelIII()
        {
            // Load training samples
            var signal = DataSet.ReadDataSet(path + "signal.dat");
            var background = DataSet.ReadDataSet(path + "background.dat");

            // Load data sample
            var data = DataSet.ReadDataSet(path + "decisionTreeData.dat");

            double[] results = new double[data.Points.Count];
            double treeweight = 0;
            
            for(int treeNum = 0; treeNum < 3; treeNum++)
            {
                var tree = new Tree();

                // Train the tree
                tree.Train(signal, background);
                int wrongClassification = 0;
                for (int i = 0; i < signal.Points.Count; ++i)
                {
                    double output = tree.RunDataPoint(signal.Points[i]);
                    if (output < 0.5)
                    {
                        wrongClassification++;
                        signal.Points[i].weight =treeweight;
                    }
                }
                for (int i = 0; i < background.Points.Count; ++i)
                {
                    double output = tree.RunDataPoint(background.Points[i]);
                    if (output < 0.5)
                    {
                        wrongClassification++;
                        background.Points[i].weight = treeweight;
                    }
                }
                for (int i = 0; i < data.Points.Count; ++i)
                {
                    double output = tree.RunDataPoint(data.Points[i]);
                    if(treeNum ==0)
                    {
                        results[i] = output;
                    }
                    else
                    {
                        results[i] += Math.Abs(output * Math.Log(Math.Abs(treeweight)));
                    }
                }
                treeweight = (1.0 - wrongClassification) / wrongClassification;
            }
            using var file = File.CreateText(path + "decisionTreeResults3.txt");
            // Calculate output value for each event and write to file
            for (int i = 0; i < results.Length; ++i)
            {
                if(results[i]>1)
                {
                    results[i] = 1;
                }
                file.WriteLine(i + "\t" + results[i]);
            }
        }
        
    }
}
