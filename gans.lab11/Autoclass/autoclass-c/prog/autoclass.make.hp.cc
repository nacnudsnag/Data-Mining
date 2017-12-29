### AUTOCLASS C MAKE FILE FOR HP-UX - bundled cc compiler

### WHEN ADDING FILES HERE, ALSO ADD THEM TO LOAD-AC ###

## THE FIRST CHARACTER OF EACH commandList must be tab
#  targetList:   dependencyList
#	commandList
## evaluate (setq-default indent-tabs-mode t)

CFLAGS = $(OSFLAGS) -Aa -O

CC = cc

DEPEND =

SRCS =	globals.c init.c io-read-data.c io-read-model.c io-results.c \
        io-results-bin.c model-expander-3.c matrix-utilities.c \
	model-single-multinomial.c model-single-normal-cm.c \
	model-single-normal-cn.c model-multi-normal-cn.c \
	model-transforms.c model-update.c search-basic.c \
        search-control.c search-control-2.c \
	search-converge.c struct-class.c struct-clsf.c \
	statistics.c predictions.c \
	struct-data.c struct-matrix.c struct-model.c \
        utils.c utils-math.c \
        intf-reports.c intf-extensions.c intf-influence-values.c \
        intf-sigma-contours.c \
	prints.c getparams.c autoclass.c

OBJS = $(SRCS:.c=.o)

autoclass:	$(OBJS) 
	$(CC) $(CFLAGS) -o autoclass $(OBJS) -lm -lc

%.o : %.c 
	$(CC) $(CFLAGS) -c $< -o $@

## depend: $(SRCS)

# IF YOU PUT ANYTHING HERE IT WILL GO AWAY

