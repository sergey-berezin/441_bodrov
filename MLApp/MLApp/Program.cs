using System;
using ClassLib;
using System.Threading.Tasks;
using System.Threading;

namespace bodrov_app
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("Type image folder name");
            string ImageFolder;
            ImageFolder = Console.ReadLine();
            var cancellationSource = new CancellationTokenSource();
            var PicPro = new PicProcessing();
            await foreach (var im in PicPro.ObjectDetecting(ImageFolder, cancellationSource))
            {
                Console.WriteLine(im);
            }


        }
    }
}
