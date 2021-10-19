using System;
using YOLOv4MLNet.DataStructures;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using YOLOv4MLNet.DataStructures;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;




namespace ClassLib
{



    public class PicProcessing
    {
        public TransformerChain<OnnxTransformer> ModelCreation ()
        {
            var mlContext = new MLContext();
            string modelPath = @"D:\University\7sem\C#\yolov4.onnx";
            // Create prediction engine
            var pipeline = mlContext.Transforms.ResizeImages(inputColumnName: "bitmap", outputColumnName: "input_1:0", imageWidth: 416, imageHeight: 416, resizing: ResizingKind.IsoPad)
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input_1:0", scaleImage: 1f / 255f, interleavePixelColors: true))
                .Append(mlContext.Transforms.ApplyOnnxModel(
                    shapeDictionary: new Dictionary<string, int[]>()
                    {
                        { "input_1:0", new[] { 1, 416, 416, 3 } },
                        { "Identity:0", new[] { 1, 52, 52, 3, 85 } },
                        { "Identity_1:0", new[] { 1, 26, 26, 3, 85 } },
                        { "Identity_2:0", new[] { 1, 13, 13, 3, 85 } },
                    },
                    inputColumnNames: new[]
                    {
                        "input_1:0"
                    },
                    outputColumnNames: new[]
                    {
                        "Identity:0",
                        "Identity_1:0",
                        "Identity_2:0"
                    },
                    modelFile: modelPath, recursionLimit: 100));

            // Fit on empty list to obtain input data schema
            var model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<YoloV4BitmapData>()));
            return model;

        }

        async public IAsyncEnumerable<string> ObjectDetecting(string ImFolder, CancellationTokenSource cancellationSource)
        {
            string[] files = Directory.GetFiles(ImFolder);
            Console.WriteLine($"ObjectDetecting: {ImFolder}");

            List<Task <IReadOnlyList<YoloV4Result>>> tasks = new List<Task<IReadOnlyList<YoloV4Result>>>();
            var model = ModelCreation();
           
            foreach (string imageName in files)
            {                
                if (!cancellationSource.Token.IsCancellationRequested)
                {
                    //Console.WriteLine($"ObjectDetecting: {imageName}");
                    Task<IReadOnlyList<YoloV4Result>> one_image = OnePictureProcessing(imageName, model);
                    tasks.Add(one_image);
                }

                else
                    break;

            }



            
            for (int i = 0; i < tasks.Count; i++)
            {
                int num = Task.WaitAny(tasks.ToArray());
                yield return PicResult(tasks[i].Result, files[i]);
                    
                    
            }
           
                  

        }


        public async Task<IReadOnlyList<YoloV4Result>> OnePictureProcessing(string filename, TransformerChain<OnnxTransformer> model)
        {
            //Console.WriteLine("OnePictureProcessing");
            return await Task.Factory.StartNew(() =>
            {
               
                string imageFolder = filename.Substring(0, filename.LastIndexOf(Path.DirectorySeparatorChar));
                var mlContext = new MLContext();
                string imageOutputFolder = Path.Combine(imageFolder, "Output");
                Directory.CreateDirectory(imageOutputFolder);
                string[] classesNames = new string[] { "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
                var predictionEngine = mlContext.Model.CreatePredictionEngine<YoloV4BitmapData, YoloV4Prediction>(model);
                //Console.WriteLine($"OnePictureProcessing: {filename}");
                using (var bitmap = new Bitmap(Image.FromFile(Path.Combine(imageFolder, filename))))
                {
                    // predict
                    var predict = predictionEngine.Predict(new YoloV4BitmapData() { Image = bitmap });
                    var results = predict.GetResults(classesNames, 0.3f, 0.7f);


                    using (var g = Graphics.FromImage(bitmap))
                    {
                        foreach (var res in results)
                        {
                            // draw predictions
                            var x1 = res.BBox[0];
                            var y1 = res.BBox[1];
                            var x2 = res.BBox[2];
                            var y2 = res.BBox[3];
                            g.DrawRectangle(Pens.Red, x1, y1, x2 - x1, y2 - y1);
                            using (var brushes = new SolidBrush(Color.FromArgb(50, Color.Red)))
                            {
                                g.FillRectangle(brushes, x1, y1, x2 - x1, y2 - y1);
                            }

                            g.DrawString(res.Label + " " + res.Confidence.ToString("0.00"),
                                         new Font("Arial", 12), Brushes.Blue, new PointF(x1, y1));
                        }
                        bitmap.Save(Path.Combine(imageOutputFolder, Path.ChangeExtension(filename.Substring(filename.LastIndexOf(Path.DirectorySeparatorChar) + 1), "_processed" + Path.GetExtension(filename))));
                    }
                    return results;
                }

            });                     
        }
    

        public string PicResult(IReadOnlyList<YoloV4Result> res, string im_name)
        {
            List<string> r = new List<string>();
            foreach (var i in res)
                r.Add(i.Label);

            
            string pic_obj = string.Join(", ", r);
            
            return $"There are {pic_obj} on {im_name}";
        }

    }


}
