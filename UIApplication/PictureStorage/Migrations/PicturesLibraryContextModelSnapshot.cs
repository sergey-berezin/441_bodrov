﻿// <auto-generated />
using System;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;
using PictureStorage;

namespace PictureStorage.Migrations
{
    [DbContext(typeof(PicturesLibraryContext))]
    partial class PicturesLibraryContextModelSnapshot : ModelSnapshot
    {
        protected override void BuildModel(ModelBuilder modelBuilder)
        {
#pragma warning disable 612, 618
            modelBuilder
                .HasAnnotation("ProductVersion", "5.0.0");

            modelBuilder.Entity("DbEntities.PictureDetails", b =>
                {
                    b.Property<int>("PictureDetailsId")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("INTEGER");

                    b.Property<byte[]>("Content")
                        .HasColumnType("BLOB");

                    b.Property<int>("PictureInfoId")
                        .HasColumnType("INTEGER");

                    b.HasKey("PictureDetailsId");

                    b.ToTable("PicturesDetails");
                });

            modelBuilder.Entity("DbEntities.PictureInformation", b =>
                {
                    b.Property<int>("Id")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("INTEGER");

                    b.Property<string>("Hash")
                        .HasColumnType("TEXT");

                    b.Property<string>("Name")
                        .HasColumnType("TEXT");

                    b.Property<int?>("PictureDetailsId")
                        .HasColumnType("INTEGER");

                    b.HasKey("Id");

                    b.HasIndex("PictureDetailsId");

                    b.ToTable("PicturesInfo");
                });

            modelBuilder.Entity("DbEntities.RecognizedCategory", b =>
                {
                    b.Property<int>("ObjectId")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("INTEGER");

                    b.Property<double>("Confidence")
                        .HasColumnType("REAL");

                    b.Property<string>("Name")
                        .HasColumnType("TEXT");

                    b.Property<int>("PictureInfoId")
                        .HasColumnType("INTEGER");

                    b.Property<int?>("PictureInformationId")
                        .HasColumnType("INTEGER");

                    b.HasKey("ObjectId");

                    b.HasIndex("PictureInformationId");

                    b.ToTable("RecognizedCategories");
                });

            modelBuilder.Entity("DbEntities.PictureInformation", b =>
                {
                    b.HasOne("DbEntities.PictureDetails", "PictureDetails")
                        .WithMany()
                        .HasForeignKey("PictureDetailsId");

                    b.Navigation("PictureDetails");
                });

            modelBuilder.Entity("DbEntities.RecognizedCategory", b =>
                {
                    b.HasOne("DbEntities.PictureInformation", null)
                        .WithMany("RecognizedCategories")
                        .HasForeignKey("PictureInformationId");
                });

            modelBuilder.Entity("DbEntities.PictureInformation", b =>
                {
                    b.Navigation("RecognizedCategories");
                });
#pragma warning restore 612, 618
        }
    }
}
