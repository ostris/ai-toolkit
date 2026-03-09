-- CreateTable
CREATE TABLE "Settings" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "key" TEXT NOT NULL,
    "value" TEXT NOT NULL
);

-- CreateTable
CREATE TABLE "Queue" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "gpu_ids" TEXT NOT NULL,
    "is_running" BOOLEAN NOT NULL DEFAULT false
);

-- CreateTable
CREATE TABLE "Job" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL,
    "gpu_ids" TEXT NOT NULL,
    "job_config" TEXT NOT NULL,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'stopped',
    "stop" BOOLEAN NOT NULL DEFAULT false,
    "return_to_queue" BOOLEAN NOT NULL DEFAULT false,
    "step" INTEGER NOT NULL DEFAULT 0,
    "info" TEXT NOT NULL DEFAULT '',
    "speed_string" TEXT NOT NULL DEFAULT '',
    "queue_position" INTEGER NOT NULL DEFAULT 0
);

-- CreateTable
CREATE TABLE "VideoDownload" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "url" TEXT NOT NULL,
    "dataset" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "progress" REAL NOT NULL DEFAULT 0,
    "error" TEXT NOT NULL DEFAULT '',
    "filename" TEXT NOT NULL DEFAULT '',
    "title" TEXT NOT NULL DEFAULT '',
    "thumbnail" TEXT NOT NULL DEFAULT '',
    "filesize" TEXT NOT NULL DEFAULT '',
    "speed" TEXT NOT NULL DEFAULT '',
    "format" TEXT NOT NULL DEFAULT '',
    "cookies_file" TEXT NOT NULL DEFAULT '',
    "pid" INTEGER,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL
);

-- CreateTable
CREATE TABLE "GalleryFolder" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "path" TEXT NOT NULL,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- CreateIndex
CREATE UNIQUE INDEX "Settings_key_key" ON "Settings"("key");

-- CreateIndex
CREATE UNIQUE INDEX "Queue_gpu_ids_key" ON "Queue"("gpu_ids");

-- CreateIndex
CREATE INDEX "Queue_gpu_ids_idx" ON "Queue"("gpu_ids");

-- CreateIndex
CREATE UNIQUE INDEX "Job_name_key" ON "Job"("name");

-- CreateIndex
CREATE INDEX "Job_status_idx" ON "Job"("status");

-- CreateIndex
CREATE INDEX "Job_gpu_ids_idx" ON "Job"("gpu_ids");

-- CreateIndex
CREATE INDEX "VideoDownload_status_idx" ON "VideoDownload"("status");

-- CreateIndex
CREATE UNIQUE INDEX "GalleryFolder_path_key" ON "GalleryFolder"("path");
