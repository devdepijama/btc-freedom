#include "Logger.cuh"

#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include <time.h>

#define PREFIX_SIZE 256

// Do not change order. It matches the values of LOGGER_LEVEL_XXXXX
static char* level_name_by_level[] = { "ERROR", "WARN", "INFO", "DEBUG" };

static void logger_print(const char *log_name, unsigned int level, unsigned int level_message, char* fmt, va_list args) {

    if (level < level_message) return;

    // Create a "prefixed" format string, by appending the log level and the log name
    char* prefixed_fmt = (char*) malloc(PREFIX_SIZE + (strlen(log_name) + 1) + (strlen(fmt) + 1));
    char* log_level = level_name_by_level[level_message];
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);

    sprintf(
        prefixed_fmt,
        "%02d/%02d/%04d %02d:%02d:%02d - [%5s] - %s - %s \n",
        tm.tm_mday, // day
        tm.tm_mon + 1, // month
        tm.tm_year + 1900, // year
        tm.tm_hour, tm.tm_min, tm.tm_sec, // hour:minutes:seconds
        log_level,
        log_name,
        fmt
    );

    vprintf(prefixed_fmt, args);

    free(prefixed_fmt);
}

Logger::Logger(char* name, unsigned int level) {
    unsigned int name_size = strlen(name) + 1;
    this->name = (char *) malloc(name_size * sizeof(char));
    strcpy(this->name, name);

    this->level = level;
}

void Logger::info(char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    logger_print(this->name, this->level, LOGGER_LEVEL_INFO, fmt, args);
    va_end(args);
}

void Logger::warn(char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    logger_print(this->name, this->level, LOGGER_LEVEL_WARN, fmt, args);
    va_end(args);
}

void Logger::error(char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    logger_print(this->name, this->level, LOGGER_LEVEL_ERROR, fmt, args);
    va_end(args);
}

void Logger::debug(char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    logger_print(this->name, this->level, LOGGER_LEVEL_DEBUG, fmt, args);
    va_end(args);
}

Logger::~Logger() {
    free(this->name);
}