#ifndef H_LOGGER
#define H_LOGGER

#define LOGGER_LEVEL_ERROR      0
#define LOGGER_LEVEL_WARN       1
#define LOGGER_LEVEL_INFO       2
#define LOGGER_LEVEL_DEBUG      3

class Logger
{
	private:
		char *name;
		unsigned int level;

	public:
		Logger(char *name, unsigned int level);

		void info(char* fmt, ...);
		void warn(char* fmt, ...);
		void error(char* fmt, ...);
		void debug(char* fmt, ...);

		~Logger();
};

#endif