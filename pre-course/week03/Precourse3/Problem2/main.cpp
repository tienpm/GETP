#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>

#define MAX_BOOK_TITLE 30
#define MAX_AUTHOR_NAME 16
#define MAX_BOOKS 2

// You CAN NOT REMOVE or CHANGE exist declaration
// But, CAN ADD whatever you want


// Total 54 bytes data
// #pragma pack(1)
typedef struct book_{
    char title[MAX_BOOK_TITLE]; // 30 bytes
    char author[MAX_AUTHOR_NAME]; // 16 bytes
    uint32_t volume_number; // 4 bytes
    uint32_t ISBN; // 4 bytes
} book;

typedef struct library_{
  book *books;
} library;

// Function to save the books data of the library
void serialize(char* path, library* lib, int number_of_books){
    FILE* wfile;
    wfile = fopen(path, "wb");
    if (wfile == NULL) {
        fprintf(stderr, "[ERROR] Open file failed\n");
        exit(1);
    }
    
    size_t flag = 0;
    flag = fwrite(lib, sizeof(lib->books[0].title) + sizeof(lib->books[0].author) + 
                  sizeof(lib->books[0].volume_number) + sizeof(lib->books[0].ISBN), number_of_books, wfile);
    if (flag == 0) {
        fprintf(stderr, "[ERROR] Open file failed\n");
        exit(1);
    }
    
    fclose(wfile);
}

// Function to load the books data of the library
void deserialize(char* path, library *lib, int number_of_books){
    FILE* rfile;
    rfile = fopen(path, "rb");
    if (rfile == NULL) {
        fprintf(stderr, "[ERROR] Open file failed\n");
        exit(1);
    }
    
    long lSize;
    // obtain file size:
    fseek (rfile , 0 , SEEK_END);
    lSize = ftell(rfile);
    rewind (rfile);

    printf("\n******************************************************\n");
    printf("Size of %s: %ld bytes\n", path, lSize);
    printf("******************************************************\n");

    for (int i = 0; i < number_of_books; i++) {
        char tmp_title[MAX_BOOK_TITLE];
        char tmp_author[MAX_AUTHOR_NAME];
        uint32_t tmp_volum_number;
        uint32_t tmp_ISBN;
        fread(&tmp_title, sizeof(tmp_title), 1, rfile);
        fread(&tmp_author, sizeof(tmp_author), 1, rfile);
        fread(&tmp_volum_number, sizeof(tmp_volum_number), 1, rfile);
        fread(&tmp_ISBN, sizeof(tmp_ISBN), 1, rfile);
        strcpy(lib->books[i].title, tmp_title);
        strcpy(lib->books[i].author, tmp_author);
        lib->books[i].volume_number = tmp_volum_number;
        lib->books[i].ISBN = tmp_ISBN;
    }

    fclose(rfile);
}

// Function to fill defaults data
// Do Not Edit This Functions
void insert_books(library *lib){

  strcpy(lib->books[0].title,"The Songs of Stardust:Harmony");
  strcpy(lib->books[0].author,"John Bolton");
  lib->books[0].ISBN = 15952557;
  lib->books[0].volume_number = 1;

  strcpy(lib->books[1].title,"Whispers Secrets are Unveiled");
  strcpy(lib->books[1].author,"donald trump");
  lib->books[1].ISBN = 67652241;
  lib->books[1].volume_number = 99;
}

// Function to print data
// Do Not Edit This Functions
void print_contents(library *lib){
  printf("\nPrint All Contents of The Library\n");
  for (int idx = 0 ; idx < MAX_BOOKS ; ++idx){
    printf("------------------------------------------------------\n");
    printf("Title : %s\n",lib->books[idx].title);
    printf("Author : %s\n",lib->books[idx].author);
    printf("ISBN : %d\n",lib->books[idx].ISBN);
    printf("volume_number : %d\n",lib->books[idx].volume_number);
    printf("------------------------------------------------------\n");
  }
}

// Do Not Edit This Functions
int main(int argc, char* argv[]){

  // Declare 2 library instance
  library lib, backup;
  void* memory_pool = malloc(54 * MAX_BOOKS * 2);

  lib.books = (book*)memory_pool;
  backup.books = (book*)((char*)memory_pool + 54 * MAX_BOOKS);

  insert_books(&lib);
  print_contents(&lib);

  // save the data of lib instance
  serialize("data.bin", &lib, MAX_BOOKS);
  
  // load the data to backup instance
  deserialize("data.bin", &backup, MAX_BOOKS);

  // Check the loaded value
  // print_contents(&lib);

  // The original value might be changed ?
  print_contents(&backup);

  return 0;
}
