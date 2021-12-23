using DbEntities;
using Microsoft.EntityFrameworkCore;

namespace PictureStorage
{
    public class PicturesLibraryContext : DbContext
    {
        public static string dbPath =
            @"D:\University\7sem\C#\441_bodrov\UIApplication\PictureStorage\pictures_lib.db";

        public DbSet<PictureInformation> PicturesInfo { get; set; }

        public DbSet<PictureDetails> PicturesDetails { get; set; }

        public DbSet<RecognizedCategory> RecognizedCategories { get; set; }

        protected override void OnConfiguring(DbContextOptionsBuilder o)
        {
            o.UseSqlite($"Data Source={dbPath}");
        }
    }
}
