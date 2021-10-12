using System;
using ClassLib;
using System.Threading.Tasks;

namespace bodrov_app
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("Type image folder name");
            string ImageFolder;
            ImageFolder = Console.ReadLine();

            var PicPro = new PicProcessing();
            await foreach (var im in PicPro.ObjectDetecting(ImageFolder))
            {
                Console.WriteLine(im);
            }


        }
    }
}
