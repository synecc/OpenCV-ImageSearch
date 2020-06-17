using System;
using System.Collections.Generic;
using OpenCvSharp;

namespace OpenCVImageSearch
{
    class ImageSearch
    {
        public static TemplateMatchModes MatchingMethod = TemplateMatchModes.CCoeffNormed;

        /// <param name="small">Small image that will be searched</param>
        /// <param name="big">Big image that small image will be searched in</param>
        /// <param name="returnCenter">middle of small bmp</param>
        public static MatchPoint Find(Mat small, Mat big, bool returnCenter = true)
        {
            MatchPoint result;
            using (Mat mresult = Match(small, big))
            {
                Cv2.MinMaxLoc(mresult, out var minVal, out var maxVal, out var minLoc, out var maxLoc);

                //For SquareDifference and SquareDifferenceNormed, the best matches are lower values. 
                //For all the other methods, the higher the better
                var matchVal = MatchingMethod == TemplateMatchModes.SqDiff ||
                               MatchingMethod == TemplateMatchModes.SqDiffNormed
                    ? minVal
                    : maxVal;
                var matchLoc = MatchingMethod == TemplateMatchModes.SqDiff ||
                               MatchingMethod == TemplateMatchModes.SqDiffNormed
                    ? minLoc
                    : maxLoc;

                if (returnCenter)
                {
                    matchLoc.X += small.Width / 2;
                    matchLoc.Y += small.Height / 2;
                }

                result = new MatchPoint(matchLoc, matchVal);
            }

            return result;
        }

        /// <summary>
        /// Finds all matches of small bmp in the big bmp
        /// </summary>
        /// <param name="small">Small image that will be searched</param>
        /// <param name="big">Big image that small image will be searched in</param>
        /// <param name="maxValue">max number of matches</param>
        /// <param name="punctuality">Must be ranged 0.00~1.00</param>
        /// <param name="returnCenter">middle of small bmp</param>
        /// <returns>List of MatchPoint based on found positions</returns>
        public static List<MatchPoint> FindAll(Mat small, Mat big, int maxValue, double punctuality,
            bool returnCenter = true)
        {
            var result = new List<MatchPoint>();

            //For SquareDifference and SquareDifferenceNormed, the best matches are lower values. 
            //For all the other methods, the higher the better
            bool boolMin = MatchingMethod == TemplateMatchModes.SqDiff ||
                           MatchingMethod == TemplateMatchModes.SqDiffNormed;

            using (Mat mresult = Match(small, big))
            {
                while (maxValue > 0)
                {
                    Cv2.MinMaxLoc(mresult, out var minVal, out var maxVal, out var minLoc, out var maxLoc);
                    var matchVal = boolMin ? minVal : maxVal;
                    bool pass = matchVal >= punctuality;
                    if (boolMin) pass = matchVal <= punctuality;
                    if (pass)
                    {
                        var matchLoc = boolMin ? minLoc : maxLoc;

                        // Fill results array with hi or lo vals, so we don't match this same location
                        Cv2.FloodFill(mresult, matchLoc, new Scalar(Convert.ToDouble(boolMin)), out var rect,
                            0.1, 1.0, FloodFillFlags.Link4);

                        if (returnCenter)
                        {
                            matchLoc.X += small.Width / 2;
                            matchLoc.Y += small.Height / 2;
                        }

                        result.Add(new MatchPoint(matchLoc, matchVal));
                    }
                    else
                    {
                        break;
                    }

                    maxValue--;
                }
            }
            return result;
        }

        private static Mat Match(Mat small, Mat big)
        {
            int rows = big.Rows - small.Rows + 1;
            int cols = big.Cols - small.Cols + 1;
            Mat result = new Mat(rows, cols, MatType.CV_32FC1, 1);
            Cv2.MatchTemplate(big, small, result, TemplateMatchModes.CCoeffNormed);
            return result;
        }
    }

    public class MatchPoint
    {
        public int X;
        public int Y;
        public double Punctuality;

        public MatchPoint() : this(0, 0, 0.0)
        {
        }

        public MatchPoint(int x, int y, double val)
        {
            X = x;
            Y = y;
            Punctuality = val;
        }

        public MatchPoint(Point pt, double val)
        {
            X = pt.X;
            Y = pt.Y;
            Punctuality = val;
        }

        public System.Drawing.Point ToPoint()
        {
            return new System.Drawing.Point(X, Y);
        }

        public Point ToCVPoint()
        {
            return new Point(X, Y);
        }
    }
}