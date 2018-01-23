#ifndef OPTION_LIST_H
#define OPTION_LIST_H
#include "list.h"

/**
 * 描述：存储键值对
 *
 * :param key:
 * :param val:
 * :param used: (default 0) 记录该键值对是否已经被读取--0:已经读取,--1:未读取；
 * */
typedef struct{
    char *key;
    char *val;
    int used;
} kvp;


int read_option(char *s, list *options);
void option_insert(list *l, char *key, char *val);
char *option_find(list *l, char *key);
int option_find_int_quiet(list *l, char *key, int def);
float option_find_float(list *l, char *key, float def);
float option_find_float_quiet(list *l, char *key, float def);
void option_unused(list *l);

#endif
