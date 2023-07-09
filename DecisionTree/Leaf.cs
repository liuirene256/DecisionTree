using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree
{
    /// <summary>
    /// A leaf or branch of a tree
    /// Does most of the work of decision trees
    /// </summary>
    internal class Leaf
    {
        /// <summary>
        /// A pointer to the next leaves, if this is a branch
        /// </summary>
        private Leaf output1 = null;
        /// <summary>
        /// A pointer to the next leaves, if this is a branch
        /// </summary>
        private Leaf output2 = null;
        /// <summary>
        /// The value of the cut that is applied at this branch (unneeded if it is a leaf)
        /// </summary>
        private double split;
        /// <summary>
        /// The index of the variable which is used to make the cut (unneeded if this is a leaf)
        /// </summary>
        private int variable;

        /// <summary>
        /// The number of background training events in this leaf
        /// </summary>
        private int nBackground = 0;
        /// <summary>
        /// The number of signal training events in this leaf
        /// </summary>
        private int nSignal = 0;

        /// <summary>
        /// A default constructor is needed for some applications, but it is generally not sensible
        /// </summary>
        internal Leaf() :
            this(-1, 0) // Default values will generate an error if used
        { }

        /// <param name="variable">The index of the variable used to make the cut</param>
        /// <param name="split">The value of the cut used for the branch</param>
        public Leaf(int variable, double split)
        {
            this.variable = variable;
            this.split = split;
        }

        /// <summary>
        /// Write the leaf to a binary file
        /// </summary>
        internal void Write(BinaryWriter bw)
        {
            bw.Write(variable);
            bw.Write(split);
            bw.Write(nSignal);
            bw.Write(nBackground);

            bw.Write(IsFinal);
            if (!IsFinal)
            {
                output1.Write(bw);
                output2.Write(bw);
            }
        }

        /// <summary>
        /// Construct a leaf from a binary file
        /// </summary>
        internal Leaf(BinaryReader br)
        {
            variable = br.ReadInt32();
            split = br.ReadDouble();
            nSignal = br.ReadInt32();
            nBackground = br.ReadInt32();

            bool fin = br.ReadBoolean();
            if (!fin)
            {
                output1 = new Leaf(br);
                output2 = new Leaf(br);
            }
        }

        /// <summary>
        /// Determines if it is a leaf or a branch (true for leaves)
        /// </summary>
        public bool IsFinal => output1 == null || output2 == null;

        /// <summary>
        /// The purity of the leaf
        /// </summary>
        public double Purity => (double)nSignal / (nSignal + nBackground);

        /// <summary>
        /// Calculates the return value for a single data point, forwarding it to other leaves as needed
        /// </summary>
        public double RunDataPoint(DataPoint dataPoint)
        {
            if (IsFinal)
            {
                return Purity;
            }

            if (DoSplit(dataPoint))
            {
                return output1.RunDataPoint(dataPoint);
            }
            else
            {
                return output2.RunDataPoint(dataPoint);
            }
        }

        /// <summary>
        /// Checks to see whether the DataPoint fails or passes the cut
        /// </summary>
        private bool DoSplit(DataPoint dataPoint)
        {
            return dataPoint.Variables[variable] <= split;
        }

        /// <summary>
        /// Trains this leaf based on input DataSets for signal and background
        /// </summary>
        public void Train(DataSet signal, DataSet background)
        {
            nSignal = signal.Points.Count;
            nBackground = background.Points.Count;

            // Determines whether this is a final leaf or if it branches
            bool branch = ChooseVariable(signal, background);

            if (branch)
            {
                // Creates a branch
                output1 = new Leaf();
                output2 = new Leaf();

                DataSet signalLeft = new DataSet(signal.Names);
                DataSet signalRight = new DataSet(signal.Names);
                DataSet backgroundLeft = new DataSet(background.Names);
                DataSet backgroundRight = new DataSet(background.Names);

                foreach (var dataPoint in signal.Points)
                {
                    if (DoSplit(dataPoint))
                    {
                        signalLeft.AddDataPoint(dataPoint);
                    }
                    else
                    {
                        signalRight.AddDataPoint(dataPoint);
                    }
                }

                foreach (var dataPoint in background.Points)
                {
                    if (DoSplit(dataPoint))
                    {
                        backgroundLeft.AddDataPoint(dataPoint);
                    }
                    else
                    {
                        backgroundRight.AddDataPoint(dataPoint);
                    }
                }

                // Trains each of the resulting leaves
                output1.Train(signalLeft, backgroundLeft);
                output2.Train(signalRight, backgroundRight);
            }
            else
            {
                //nsole.WriteLine(Purity);
            }
            // Do nothing more if it is not a branch
        }

        /// <summary>
        /// Chooses which variable and cut value to use
        /// </summary>
        /// <returns>True if a branch was created, false if this is a final leaf</returns>

        private bool ChooseVariable(DataSet signal, DataSet background)
        {
            double bestPurity = 1;
            double pLeft = 1;
            double pRight = 1;
            double right = 0;
            double left = 0;
            double tRight = 0;
            double tLeft = 0;

            if (signal.Points.Count < 50 || signal.Points.Count < 50)
            {
                return false;
            }

            for (int var = 0; var < signal.Points[0].Variables.Length; var++)
            {
                int i = 0;

                double sLeft = 0;
                double sRight = 0;
                double bLeft = 0;
                double bRight = 0;

                double[] splits = new double[signal.Points.Count + background.Points.Count];
                int[] label = new int[signal.Points.Count + background.Points.Count];
                foreach (DataPoint point in signal.Points) //loops through points
                {
                    splits[i] = point.Variables[var];
                    label[i] = i;
                    sRight+= point.weight;
                    i++;
                }
                foreach (DataPoint point in background.Points) //loops through points
                {
                    splits[i] = point.Variables[var];
                    label[i] = i;
                    bRight+= point.weight;
                    i++;
                }
                Array.Sort(splits, label);

                double purityLeft = 0;
                double purityRight = 0;
                for (int ind = 0; ind < splits.GetLength(0) - 1; ind++)
                {
                    if (label[ind] <signal.Points.Count)
                    {
                        sRight-=signal.Points[label[ind]].weight;
                        sLeft+= signal.Points[label[ind]].weight;
                    }
                    else
                    {
                        bRight -= background.Points[label[ind] - signal.Points.Count].weight;
                        bLeft += background.Points[label[ind] - signal.Points.Count].weight;
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
                        variable = var;
                        bestPurity = purityLeft + purityRight;
                        split = (splits[ind] + splits[ind + 1]) / 2;
                        pLeft = purityLeft;
                        pRight = purityRight;
                        tLeft = left;
                        tRight = right;
                    }
                }
            }
            // Console.WriteLine(variable + " split: " + split + " " + right + " " + left + " " + right2 + " " + left2 + " " + bestPurity);

            //nsole.WriteLine(variable + " split: " + split);
            if (tLeft < 50 || tRight < 50)
            {
                return false;
            }
            if (Purity > .5)
            {
                if (1 - Purity < pLeft && 1 - Purity < pRight)
                {

                    return false;
                }
            }
            else
            {
                if (Purity < pLeft && Purity < pRight)
                {
                    return false;
                }
            }

          //Console.WriteLine(variable + " split: " + split);
            return true;

        }

        //private bool ChooseVariable(DataSet signal, DataSet background)
        //{
        //    double bestPurity = 1;
        //    double pLeft = 1;
        //    double pRight = 1;
        //    double right = 0;
        //    double left = 0;
        //    double tRight = 0;
        //    double tLeft = 0;

        //    if(signal.Points.Count < 50 || signal.Points.Count < 50)
        //    {
        //        return false;
        //    }

        //    for (int var = 0; var < signal.Points[0].Variables.Length; var++)
        //    {
        //        int i = 0;

        //        double sLeft = 0;
        //        double sRight = 0;
        //        double bLeft = 0;
        //        double bRight = 0;

        //        double[] splits = new double[signal.Points.Count + background.Points.Count];
        //        double[] label = new double[signal.Points.Count + background.Points.Count];
        //        foreach (DataPoint point in signal.Points) //loops through points
        //        {
        //            splits[i] = point.Variables[var];
        //            label[i] = 1;
        //            sRight++;
        //            i++;
        //        }
        //        foreach (DataPoint point in background.Points) //loops through points
        //        {
        //            splits[i] = point.Variables[var];
        //            label[i] = 0;
        //            bRight++;
        //            i++;
        //        }
        //        Array.Sort(splits, label);

        //        double purityLeft = 0;
        //        double purityRight = 0;
        //        for (int ind = 0; ind < splits.GetLength(0) - 1; ind++)
        //        {
        //            if (label[ind] == 0)
        //            {
        //                bRight--;
        //                bLeft++;
        //            }
        //            else
        //            {
        //                sRight--;
        //                sLeft++;
        //            }
        //            left = sLeft + bLeft;
        //            right = sRight + bRight;

        //            purityLeft = sLeft / left;
        //            purityRight = sRight / right;

        //            if (purityLeft > 0.5)
        //            {
        //                purityLeft = 1 - purityLeft;
        //            }
        //            else if (purityRight > 0.5)
        //            {
        //                purityRight = 1 - purityRight;
        //            }
        //            else
        //            {
        //                purityLeft = 2;
        //            }
        //            if (purityLeft + purityRight < bestPurity && left >=50 && right >= 50)
        //            {
        //                variable = var;
        //                bestPurity = purityLeft + purityRight;
        //                split = (splits[ind] + splits[ind + 1]) / 2;
        //                pLeft = purityLeft;
        //                pRight = purityRight;
        //                tLeft = left;
        //                tRight = right;
        //            }
        //        }
        //    }
        //    // Console.WriteLine(variable + " split: " + split + " " + right + " " + left + " " + right2 + " " + left2 + " " + bestPurity);

        //    Console.WriteLine(variable + " split: " + split);
        //    if(tLeft < 50 || tRight < 50)
        //    {
        //        return false;
        //    }
        //    if (Purity>.5)
        //    {
        //        if(1-Purity<pLeft && 1-Purity<pRight )
        //        {

        //            return false;
        //        }
        //    }
        //    else
        //    {
        //        if(Purity<pLeft && Purity < pRight)
        //        {
        //            return false;
        //        }
        //    }

        //    Console.WriteLine(variable + " split: " + split);
        //    return true;

        //}
    }
}
