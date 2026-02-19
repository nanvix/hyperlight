// Included from hyperlight_guest_capi/include
#include "hyperlight_guest.h"
// Included from hyperlight_guest_bin/third_party/libc
#include "stdint.h"
#include "string.h"
#include "stdlib.h"
#include "assert.h"
// Included from hyperlight_guest_bin/third_party/printf
#include "printf.h"

#define GUEST_SCRATCH_SIZE (0x40000) // default scratch size
#define MAX_BUFFER_SIZE (1024)

// Buffer sizes for FS operations
#define FS_DIR_BUFFER_SIZE (4096)   // Buffer for directory listing results
#define FS_PATH_BUFFER_SIZE (256)   // Buffer for path strings (CWD, etc.)

static char big_array[1024 * 1024] = {0};

const char *echo(const char *str) { return str; }

float echo_float(float f) { return f; }

double echo_double(double d) { return d; }

hl_Vec *set_byte_array_to_zero(const hl_FunctionCall* params) {
  hl_Vec input = params->parameters[0].value.VecBytes;
  uint8_t *x = malloc(input.len);
  for (uintptr_t i = 0; i < input.len; i++) {
    x[i] = 0;
  }
  return hl_flatbuffer_result_from_Bytes(x, input.len);
}

int print_output(const char *message) {
  int res = printf("%s", message);
  return res;
}

__attribute__((optnone)) 
int stack_allocate(int32_t length) {
  void *buffer = alloca(length);
  (void)buffer;

  return length;
}

__attribute__((optnone)) 
void stack_overflow_helper(int32_t i) {
  if (i == 0) {
    return;
  }
  char nums[16384] = {i};
  (void)nums;

  stack_overflow_helper(i - 1);
}

__attribute__((optnone)) 
int stack_overflow(int32_t i) {
  stack_overflow_helper(i);

  return i;
}

int buffer_overrun(const char *String) {
  char buffer[17];
  (void)buffer;
  int length = strlen(String);

  if (length > 0) {
    strncpy(buffer, String, length);
  }
  int result = (int)(17 - length);

  return result;
}

__attribute__((optnone)) 
int large_var(void) {
  char buffer[GUEST_SCRATCH_SIZE + 1] = {0};
  (void)buffer;

  return GUEST_SCRATCH_SIZE;
}

int small_var(void) {
  char buffer[1024] = {0};
  (void)buffer;

  return 1024;
}

int call_malloc(int32_t size) {
  void *heap_memory = malloc(size);
  if (NULL == heap_memory) {
    hl_set_error(hl_ErrorCode_GuestError, "Malloc Failed");
  }

  return size;
}

int malloc_and_free(int32_t size) {
  void *heap_memory = malloc(size);
  if (NULL == heap_memory) {
    hl_set_error(hl_ErrorCode_GuestError, "Malloc Failed");
  }

  free(heap_memory);

  return size;
}

int print_two_args(const char *arg1, int32_t arg2) {
  int result = printf("Message: arg1:%s arg2:%d.", arg1, arg2);

  return result;
}

int print_three_args(const char *arg1, int32_t arg2, int64_t arg3) {
  int result = printf("Message: arg1:%s arg2:%d arg3:%d.", arg1, arg2, arg3);

  return result;
}

 int print_four_args(const char *arg1, int32_t arg2, int64_t arg3,
                        const char *arg4) {
  int result = printf("Message: arg1:%s arg2:%d arg3:%d arg4:%s.", arg1, arg2,
                      arg3, arg4);

  return result;
}

 int print_five_args(const char *arg1, int32_t arg2, int64_t arg3,
                        const char *arg4, const char *arg5) {
  int result = printf("Message: arg1:%s arg2:%d arg3:%d arg4:%s arg5:%s.", arg1,
                      arg2, arg3, arg4, arg5);

  return result;
}

 int print_six_args(const char *arg1, int32_t arg2, int64_t arg3,
                       const char *arg4, const char *arg5, bool arg6) {
  int result =
      printf("Message: arg1:%s arg2:%d arg3:%d arg4:%s arg5:%s arg6:%s.", arg1,
             arg2, arg3, arg4, arg5, arg6 ? "true" : "false");

  return result;
}

 int print_seven_args(const char *arg1, int32_t arg2, int64_t arg3,
                         const char *arg4, const char *arg5, bool arg6,
                         bool arg7) {
  int result = printf(
      "Message: arg1:%s arg2:%d arg3:%d arg4:%s arg5:%s arg6:%s arg7:%s.", arg1,
      arg2, arg3, arg4, arg5, arg6 ? "true" : "false", arg7 ? "true" : "false");

  return result;
}

 int print_eight_args(const char *arg1, int32_t arg2, int64_t arg3,
                         const char *arg4, const char *arg5, bool arg6,
                         bool arg7, uint32_t arg8) {
  int result = printf("Message: arg1:%s arg2:%d arg3:%d arg4:%s arg5:%s "
                      "arg6:%s arg7:%s arg8:%d.",
                      arg1, arg2, arg3, arg4, arg5, arg6 ? "true" : "false",
                      arg7 ? "true" : "false", arg8);

  return result;
}

 int print_nine_args(const char *arg1, int32_t arg2, int64_t arg3,
                        const char *arg4, const char *arg5, bool arg6,
                        bool arg7, uint32_t arg8, uint64_t arg9) {
  int result = printf("Message: arg1:%s arg2:%d arg3:%d arg4:%s arg5:%s "
                      "arg6:%s arg7:%s arg8:%d arg9:%d.",
                      arg1, arg2, arg3, arg4, arg5, arg6 ? "true" : "false",
                      arg7 ? "true" : "false", arg8, arg9);

  return result;
}

 int print_ten_args(const char *arg1, int32_t arg2, int64_t arg3,
                       const char *arg4, const char *arg5, bool arg6, bool arg7,
                       uint32_t arg8, uint64_t arg9, int32_t arg10) {
  int result = printf("Message: arg1:%s arg2:%d arg3:%d arg4:%s arg5:%s "
                      "arg6:%s arg7:%s arg8:%d arg9:%d arg10:%d.",
                      arg1, arg2, arg3, arg4, arg5, arg6 ? "true" : "false",
                      arg7 ? "true" : "false", arg8, arg9, arg10);

  return result;
}

 int print_eleven_args(const char *arg1, int32_t arg2, int64_t arg3,
                          const char *arg4, const char *arg5, bool arg6,
                          bool arg7, uint32_t arg8, uint64_t arg9,
                          int32_t arg10, float arg11) {
  int result = printf("Message: arg1:%s arg2:%d arg3:%d arg4:%s arg5:%s "
                      "arg6:%s arg7:%s arg8:%d arg9:%d arg10:%d arg11:%.3f.",
                      arg1, arg2, arg3, arg4, arg5, arg6 ? "true" : "false",
                      arg7 ? "true" : "false", arg8, arg9, arg10, arg11);

  return result;
}

int set_static(void) {
  int length = sizeof(big_array);
  for (int l = 0; l < length; l++) {
    big_array[l] = l;
  }
  return length;
}

hl_Vec *get_size_prefixed_buffer(const hl_FunctionCall* params) {
  hl_Vec input = params->parameters[0].value.VecBytes;
  return hl_flatbuffer_result_from_Bytes(input.data, input.len);
}

int guest_abort_with_code(int32_t code) {
  hl_abort_with_code(code);
  return -1;
}

int guest_abort_with_msg(int32_t code, const char *message) {
  hl_abort_with_code_and_message(code, message);
  return -1;
}

int execute_on_stack(void) {
  uint8_t hlt = 0xF4;
  ((void (*)()) & hlt)();
  return -1;
}

int log_message(const char *message, int64_t level) {
  LOG((hl_Level)level, message);
  return -1;
}

hl_Vec *twenty_four_k_in_eight_k_out(const hl_FunctionCall* params) {
  hl_Vec input = params->parameters[0].value.VecBytes;
  assert(input.len == 24 * 1024);
  return hl_flatbuffer_result_from_Bytes(input.data, 8 * 1024);
}

int guest_function(const char *from_host) {
  char guest_message[256] = "Hello from GuestFunction1, ";
  int len = strlen(from_host);
  strncat(guest_message, from_host, len);

  hl_Parameter params = {.tag = hl_ParameterType_String,
                         .value = {.String = guest_message}};
  const hl_FunctionCall host_call = {.function_name = "HostMethod1",
                                     .parameters = &params,
                                     .parameters_len = 1,
                                     .return_type = hl_ReturnType_Int};
  hl_call_host_function(&host_call);
  hl_get_host_return_value_as_Int();

  return 0;
}

bool guest_fn_checks_if_host_returns_bool_value(int32_t a, int32_t b) {
  hl_Parameter params[2];

  params[0].tag = hl_ParameterType_Int;
  params[0].value.Int = a;

  params[1].tag = hl_ParameterType_Int;
  params[1].value.Int = b;

  const hl_FunctionCall host_call = {.function_name = "HostBool",
                                     .parameters = params,
                                     .parameters_len = 2,
                                     .return_type = hl_ReturnType_Bool
                                    };
  hl_call_host_function(&host_call);                                 
  return hl_get_host_return_value_as_Bool();
}

float guest_fn_checks_if_host_returns_float_value(float a, float b) {
  hl_Parameter params[2];

  params[0].tag = hl_ParameterType_Float;
  params[0].value.Float = a;

  params[1].tag = hl_ParameterType_Float;
  params[1].value.Float = b;

  const hl_FunctionCall host_call = {.function_name = "HostAddFloat",
                                     .parameters = params,
                                     .parameters_len = 2,
                                     .return_type = hl_ReturnType_Float
                                    };
  hl_call_host_function(&host_call); 
  return hl_get_host_return_value_as_Float();
}

double guest_fn_checks_if_host_returns_double_value(double a, double b) {
  hl_Parameter params[2];

  params[0].tag = hl_ParameterType_Double;
  params[0].value.Double = a;

  params[1].tag = hl_ParameterType_Double;
  params[1].value.Double = b;

  const hl_FunctionCall host_call = {.function_name = "HostAddDouble",
                                     .parameters = params,
                                     .parameters_len = 2,
                                     .return_type = hl_ReturnType_Double
                                    };
  hl_call_host_function(&host_call); 
  return hl_get_host_return_value_as_Double();
}

const char* guest_fn_checks_if_host_returns_string_value() {
  char guest_message[256] = "Guest Function";
  hl_Parameter params;

  params.tag = hl_ParameterType_String;
  params.value.String = guest_message;

  const hl_FunctionCall host_call = {.function_name = "HostAddStrings",
                                     .parameters = &params,
                                     .parameters_len = 1,
                                     .return_type = hl_ReturnType_String
                                    };
  hl_call_host_function(&host_call); 
  return hl_get_host_return_value_as_String();
}

// Simple LCG random number generator
static uint64_t lcg_state = 0;

static uint64_t rdtsc(void) {
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static uint64_t lcg_next(void) {
    lcg_state = lcg_state * 6364136223846793005ULL + 1;
    return lcg_state;
}

// RandomReadChunks - reads 10 random 256-byte chunks from a file
// Returns: Vec<u8> where each sample is 8 bytes offset + 256 bytes data
// Total: 10 * 264 = 2640 bytes
#define NUM_SAMPLES 10
#define CHUNK_SIZE 256
#define SAMPLE_SIZE (8 + CHUNK_SIZE)

hl_Vec *random_read_chunks(const hl_FunctionCall *params) {
    const char *path = params->parameters[0].value.String;
    
    // Check if FS is initialized
    if (!hl_fs_initialized()) {
        return hl_flatbuffer_result_from_Bytes(NULL, 0);
    }
    
    // Open file
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return hl_flatbuffer_result_from_Bytes(NULL, 0);
    }
    
    // Get file size via lseek
    int64_t file_size = lseek(fd, 0, SEEK_END);
    if (file_size < 0) {
        close(fd);
        return hl_flatbuffer_result_from_Bytes(NULL, 0);
    }
    
    // Allocate result buffer
    uint8_t *result = malloc(NUM_SAMPLES * SAMPLE_SIZE);
    if (!result) {
        close(fd);
        return hl_flatbuffer_result_from_Bytes(NULL, 0);
    }
    
    // Seed RNG with RDTSC
    lcg_state = rdtsc();
    
    // Max valid offset
    uint64_t max_offset = (file_size > CHUNK_SIZE) ? (file_size - CHUNK_SIZE) : 0;
    
    for (int i = 0; i < NUM_SAMPLES; i++) {
        // Get random offset
        uint64_t random_val = lcg_next();
        uint64_t offset = (max_offset > 0) ? (random_val % max_offset) : 0;
        
        // Seek to offset
        if (lseek(fd, (int64_t)offset, SEEK_SET) < 0) {
            free(result);
            close(fd);
            return hl_flatbuffer_result_from_Bytes(NULL, 0);
        }
        
        // Write offset to result (little-endian)
        int sample_start = i * SAMPLE_SIZE;
        for (int j = 0; j < 8; j++) {
            result[sample_start + j] = (offset >> (j * 8)) & 0xFF;
        }
        
        // Read chunk data
        int data_start = sample_start + 8;
        int64_t bytes_read = read(fd, &result[data_start], CHUNK_SIZE);
        if (bytes_read < 0) {
            free(result);
            close(fd);
            return hl_flatbuffer_result_from_Bytes(NULL, 0);
        }
    }
    
    close(fd);
    return hl_flatbuffer_result_from_Bytes(result, NUM_SAMPLES * SAMPLE_SIZE);
}

// Returns 1 if HyperlightFS is initialized, 0 otherwise
int is_fs_initialized(void) {
    return hl_fs_initialized() ? 1 : 0;
}

// Reads a file from HyperlightFS and returns its contents.
// Returns empty vec if file not found or FS not initialized.
hl_Vec *read_file(const hl_FunctionCall *params) {
    const char *path = params->parameters[0].value.String;
    
    // Return empty if FS not initialized
    if (!hl_fs_initialized()) {
        return hl_flatbuffer_result_from_Bytes(NULL, 0);
    }
    
    // Open file
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return hl_flatbuffer_result_from_Bytes(NULL, 0);
    }
    
    // Get file size via lseek
    int64_t file_size = lseek(fd, 0, SEEK_END);
    if (file_size < 0) {
        close(fd);
        return hl_flatbuffer_result_from_Bytes(NULL, 0);
    }
    
    // Seek back to start
    if (lseek(fd, 0, SEEK_SET) < 0) {
        close(fd);
        return hl_flatbuffer_result_from_Bytes(NULL, 0);
    }
    
    // Allocate buffer for file contents
    uint8_t *buffer = malloc((size_t)file_size);
    if (!buffer) {
        close(fd);
        return hl_flatbuffer_result_from_Bytes(NULL, 0);
    }
    
    // Read entire file
    int64_t bytes_read = read(fd, buffer, (size_t)file_size);
    close(fd);
    
    if (bytes_read < 0) {
        free(buffer);
        return hl_flatbuffer_result_from_Bytes(NULL, 0);
    }
    
    return hl_flatbuffer_result_from_Bytes(buffer, (size_t)bytes_read);
}

// =============================================================================
// FAT Filesystem Functions (for integration tests)
// =============================================================================

// Writes content to a file on a FAT mount.
// Returns 1 on success, 0 on error.
hl_Vec *write_fat_file(const hl_FunctionCall *params) {
    const char *path = params->parameters[0].value.String;
    hl_Vec content = params->parameters[1].value.VecBytes;
    
    if (!hl_fs_initialized()) {
        return hl_flatbuffer_result_from_Bool(false);
    }
    
    // Open file for writing, create if doesn't exist, truncate
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC);
    if (fd < 0) {
        return hl_flatbuffer_result_from_Bool(false);
    }
    
    // Write content
    int64_t written = write(fd, content.data, content.len);
    close(fd);
    
    return hl_flatbuffer_result_from_Bool(written == (int64_t)content.len);
}

// Reads a file from a FAT mount and returns its contents.
// Returns empty vec on error.
hl_Vec *read_fat_file(const hl_FunctionCall *params) {
    const char *path = params->parameters[0].value.String;
    
    if (!hl_fs_initialized()) {
        return hl_flatbuffer_result_from_Bytes(NULL, 0);
    }
    
    // Open file for reading
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return hl_flatbuffer_result_from_Bytes(NULL, 0);
    }
    
    // Get file size via lseek
    int64_t file_size = lseek(fd, 0, SEEK_END);
    if (file_size < 0) {
        close(fd);
        return hl_flatbuffer_result_from_Bytes(NULL, 0);
    }
    
    // Seek back to start
    if (lseek(fd, 0, SEEK_SET) < 0) {
        close(fd);
        return hl_flatbuffer_result_from_Bytes(NULL, 0);
    }
    
    // Allocate buffer for file contents
    uint8_t *buffer = malloc((size_t)file_size);
    if (!buffer) {
        close(fd);
        return hl_flatbuffer_result_from_Bytes(NULL, 0);
    }
    
    // Read entire file
    int64_t bytes_read = read(fd, buffer, (size_t)file_size);
    close(fd);
    
    if (bytes_read < 0) {
        free(buffer);
        return hl_flatbuffer_result_from_Bytes(NULL, 0);
    }
    
    return hl_flatbuffer_result_from_Bytes(buffer, (size_t)bytes_read);
}

// Deletes a file from a FAT mount.
// Returns 1 on success, 0 on error.
int delete_fat_file(const char *path) {
    if (!hl_fs_initialized()) {
        return 0;
    }
    return unlink(path) == 0 ? 1 : 0;
}

// Creates a directory on a FAT mount.
// Returns 1 on success, 0 on error.
int mkdir_fat(const char *path) {
    if (!hl_fs_initialized()) {
        return 0;
    }
    return mkdir(path, 0755) == 0 ? 1 : 0;
}

// Removes an empty directory from a FAT mount.
// Returns 1 on success, 0 on error.
int rmdir_fat(const char *path) {
    if (!hl_fs_initialized()) {
        return 0;
    }
    return rmdir(path) == 0 ? 1 : 0;
}

// Lists directory contents on a FAT mount.
// Returns entry names as a newline-separated string.
const char *list_dir_fat(const char *path) {
    static char dir_buf[FS_DIR_BUFFER_SIZE];
    if (!hl_fs_initialized()) {
        return "";
    }
    int64_t result = hl_fs_readdir(path, dir_buf, sizeof(dir_buf));
    if (result < 0) {
        return "";
    }
    return dir_buf;
}

// Renames a file or directory on a FAT mount.
// Returns 1 on success, 0 on error.
int rename_fat(const char *oldpath, const char *newpath) {
    if (!hl_fs_initialized()) {
        return 0;
    }
    return rename(oldpath, newpath) == 0 ? 1 : 0;
}

// Gets the current working directory.
// Returns pointer to static buffer with CWD, or empty string on error.
const char *get_cwd(void) {
    static char cwd_buffer[FS_PATH_BUFFER_SIZE];
    if (!hl_fs_initialized()) {
        return "";
    }
    if (getcwd(cwd_buffer, sizeof(cwd_buffer)) == NULL) {
        return "";
    }
    return cwd_buffer;
}

// Changes the current working directory.
// Returns 1 on success, 0 on error.
int do_chdir(const char *path) {
    if (!hl_fs_initialized()) {
        return 0;
    }
    return chdir(path) == 0 ? 1 : 0;
}



// Gets file size on FAT mount.
// Returns -1 on error, otherwise the file size.
int64_t stat_fat_size(const char *path) {
    if (!hl_fs_initialized()) {
        return -1;
    }
    hl_Stat st;
    if (stat(path, &st) < 0) {
        return -1;
    }
    return (int64_t)st.size;
}

// Checks if a path exists on FAT mount.
// Returns 1 for file, 2 for directory, 0 for not found, -1 for error.
int exists_fat(const char *path) {
    if (!hl_fs_initialized()) {
        return -1;
    }
    hl_Stat st;
    if (stat(path, &st) < 0) {
        return 0;  // Not found
    }
    return st.is_dir ? 2 : 1;
}

// ============================================================================
// Guest-Created FAT Mount Tests (dynamic filesystem creation)
// ============================================================================

// Creates a guest-side FAT mount at the specified path.
// Returns 1 (true) on success, 0 (false) on error.
int create_fat_mount(const char *path, int64_t size_bytes) {
    return hl_fs_create_fat_mount(path, size_bytes) == 0 ? 1 : 0;
}

// Unmounts a guest-created FAT mount.
// Returns 1 (true) on success, 0 (false) on error.
int unmount_fat(const char *path) {
    return hl_fs_unmount_fat(path) == 0 ? 1 : 0;
}

// Checks if a mount was created by the guest.
// Returns 1 (true) if guest-created, 0 (false) otherwise.
int is_guest_created_mount(const char *path) {
    return hl_fs_is_guest_created_mount(path);
}

// Full test: create mount, write, read, delete, unmount.
// Returns 1 on success, negative error code on failure.
int test_guest_fat_full_cycle(const char *mount_path, int64_t size_bytes) {
    // Create mount
    if (hl_fs_create_fat_mount(mount_path, size_bytes) != 0) {
        return -1;  // Failed to create mount
    }

    // Verify it's guest-created
    if (!hl_fs_is_guest_created_mount(mount_path)) {
        hl_fs_unmount_fat(mount_path);
        return -2;  // Should be guest-created
    }

    // Build file path
    char file_path[256];
    snprintf(file_path, sizeof(file_path), "%s/test.txt", mount_path);

    // Write a file
    int fd = open(file_path, O_CREAT | O_WRONLY);
    if (fd < 0) {
        hl_fs_unmount_fat(mount_path);
        return -3;  // Failed to open file
    }
    const char *content = "Hello from C guest!";
    int64_t written = write(fd, content, strlen(content));
    close(fd);
    if (written != (int64_t)strlen(content)) {
        hl_fs_unmount_fat(mount_path);
        return -4;  // Write failed
    }

    // Read it back
    fd = open(file_path, O_RDONLY);
    if (fd < 0) {
        hl_fs_unmount_fat(mount_path);
        return -5;  // Failed to open for read
    }
    char read_buf[64] = {0};
    int64_t bytes_read = read(fd, read_buf, sizeof(read_buf) - 1);
    close(fd);
    if (bytes_read != (int64_t)strlen(content)) {
        hl_fs_unmount_fat(mount_path);
        return -6;  // Read wrong amount
    }
    if (strcmp(read_buf, content) != 0) {
        hl_fs_unmount_fat(mount_path);
        return -7;  // Content mismatch
    }

    // Delete the file
    if (unlink(file_path) != 0) {
        hl_fs_unmount_fat(mount_path);
        return -8;  // Delete failed
    }

    // Verify it's gone
    hl_Stat st;
    if (stat(file_path, &st) == 0) {
        hl_fs_unmount_fat(mount_path);
        return -9;  // File should not exist
    }

    // Unmount
    if (hl_fs_unmount_fat(mount_path) != 0) {
        return -10;  // Unmount failed
    }

    // Verify no longer guest-created
    if (hl_fs_is_guest_created_mount(mount_path)) {
        return -11;  // Should no longer be guest-created after unmount
    }

    return 1;  // Success
}

// Test unmounting with open files - should fail when files are still open.
// Returns 1 on success (correct behavior), negative error code on failure.
int test_unmount_with_open_file(const char *mount_path, int64_t size_bytes) {
    // Create mount
    if (hl_fs_create_fat_mount(mount_path, size_bytes) != 0) {
        return -1;  // Failed to create mount
    }

    // Build file path
    char file_path[256];
    snprintf(file_path, sizeof(file_path), "%s/test.txt", mount_path);

    // Open a file and keep it open
    int fd = open(file_path, O_CREAT | O_WRONLY);
    if (fd < 0) {
        hl_fs_unmount_fat(mount_path);
        return -2;  // Failed to open file
    }

    // Write something
    const char *content = "test";
    write(fd, content, strlen(content));

    // Try to unmount while file is open - this SHOULD FAIL
    int unmount_result = hl_fs_unmount_fat(mount_path);
    if (unmount_result == 0) {
        // BUG: Unmount should have failed!
        close(fd);
        return -3;  // Unmount succeeded but should have failed
    }

    // Good - unmount correctly rejected. Now close the file.
    close(fd);

    // Now unmount should succeed
    if (hl_fs_unmount_fat(mount_path) != 0) {
        return -4;  // Unmount should succeed after close
    }

    return 1;  // Success - correct behavior
}

// ============================================================================
// New C API test functions (opendir/readdir/closedir, access, openat, fcntl)
// ============================================================================

// Test opendir/readdir/closedir - returns entry count or -1 on error
int test_opendir_readdir(const char *path) {
    if (!hl_fs_initialized()) {
        return -1;
    }

    hl_hl_DIR *dir = opendir(path);
    if (dir == NULL) {
        return -1;
    }

    int count = 0;
    hl_hl_dirent_t *entry;
    while ((entry = readdir(dir)) != NULL) {
        count++;
    }

    if (closedir(dir) != 0) {
        return -1;
    }

    return count;
}

// Test opendir and list entries - returns names as newline-separated string
const char *test_opendir_list(const char *path) {
    static char result[FS_DIR_BUFFER_SIZE];
    result[0] = '\0';

    if (!hl_fs_initialized()) {
        return "";
    }

    hl_hl_DIR *dir = opendir(path);
    if (dir == NULL) {
        return "";
    }

    char *ptr = result;
    char *end = result + sizeof(result) - 2;

    hl_hl_dirent_t *entry;
    while ((entry = readdir(dir)) != NULL && ptr < end) {
        // Append "D:" or "F:" prefix based on type
        if (ptr < end) {
            *ptr++ = (entry->d_type == hl_HL_DT_DIR) ? 'D' : 'F';
        }
        if (ptr < end) {
            *ptr++ = ':';
        }
        // Copy name
        const char *name = entry->d_name;
        while (*name && ptr < end) {
            *ptr++ = *name++;
        }
        if (ptr < end) {
            *ptr++ = '\n';
        }
    }
    *ptr = '\0';

    closedir(dir);
    return result;
}

// Test access() function
// Returns result of access(path, mode) 
int test_access(const char *path, int mode) {
    if (!hl_fs_initialized()) {
        return hl_HL_ENOTSUP;
    }
    return access(path, mode);
}

// Test openat with AT_FDCWD - should work like open()
int test_openat_cwd(const char *path, int flags) {
    if (!hl_fs_initialized()) {
        return hl_HL_ENOTSUP;
    }
    int fd = openat(hl_HL_AT_FDCWD, path, flags);
    if (fd >= 0) {
        close(fd);
        return 1;  // Success
    }
    return fd;  // Error code
}

// ============================================================================
// FD LIFETIME REGRESSION TEST
// ============================================================================
// This test catches a bug where the Rust File object would drop (and close
// the fd) when hl_fs_open() returned, because File owns the fd via Drop.
// The fix was to use into_raw_fd() to transfer ownership to the caller.

// Helper: open a file and return the fd (simulates real-world usage)
static int open_and_return_fd(const char *path) {
    return open(path, O_RDONLY);
}

// Test that fd survives being returned from a function
// Returns: 1 = success, -1 = open failed, -2 = read failed after return
int test_fd_lifetime_across_function_return(const char *path) {
    if (!hl_fs_initialized()) {
        return -99;
    }

    // Get fd from helper function (fd must survive the return!)
    int fd = open_and_return_fd(path);
    if (fd < 0) {
        return -1;  // Open failed
    }

    // Try to use the fd - this would fail if fd was closed on return
    char buf[16];
    int64_t n = read(fd, buf, sizeof(buf));
    close(fd);

    return (n >= 0) ? 1 : -2;  // 1 = success, -2 = read failed
}

// Test fcntl F_GETFL/F_SETFL
int test_fcntl_flags(const char *path) {
    if (!hl_fs_initialized()) {
        return -1;
    }

    // Open a file
    int fd = open(path, hl_HL_O_RDONLY);
    if (fd < 0) {
        return -1;
    }

    // Get flags
    int flags = fcntl(fd, hl_HL_F_GETFL, 0);
    if (flags < 0) {
        close(fd);
        return -2;
    }

    // Set flags (should succeed even if no-op)
    int result = fcntl(fd, hl_HL_F_SETFL, flags);
    close(fd);

    if (result < 0) {
        return -3;
    }

    return flags;  // Return original flags
}

// Test dup - verify duplicated fd works and shares file position
// Returns true if dup works correctly, false otherwise.
bool test_dup(const char *path) {
    if (!hl_fs_initialized()) {
        return false;  // Can't test without FS
    }

    // Open the file
    int fd = open(path, hl_HL_O_RDONLY);
    if (fd < 0) {
        return false;  // Can't open file to test
    }

    // Duplicate the fd
    int fd2 = dup(fd);
    if (fd2 < 0) {
        close(fd);
        return false;  // dup() failed
    }

    // fd2 should be a different fd number
    if (fd == fd2) {
        close(fd);
        close(fd2);
        return false;  // fd and fd2 should be different
    }

    // Read one byte from fd to advance position
    char buf[2];
    if (read(fd, buf, 1) != 1) {
        close(fd);
        close(fd2);
        return false;  // Read failed
    }

    // Both RO and FAT files share position per POSIX
    // Read from fd2 - position is shared so we continue from where fd left off
    char buf2[2];
    int64_t n = read(fd2, buf2, 1);
    
    // Close both fds
    close(fd);
    close(fd2);

    // Both should have read successfully
    return n >= 0;
}

// Test dup2 - verify dup2 to specific fd works
// Returns true if dup2 works correctly, false otherwise.
bool test_dup2(const char *path) {
    if (!hl_fs_initialized()) {
        return false;  // Can't test without FS
    }

    int fd = open(path, hl_HL_O_RDONLY);
    if (fd < 0) {
        return false;  // Can't open file to test
    }

    // Use fd 100 as target - arbitrary high value unlikely to conflict
    const int target_fd = 100;
    int result = dup2(fd, target_fd);
    if (result != target_fd) {
        close(fd);
        return false;  // dup2() should return target_fd on success
    }

    // Verify the duplicated fd can read
    char buf[16];
    int64_t n = read(target_fd, buf, sizeof(buf) - 1);
    
    // Close both fds
    close(fd);
    close(target_fd);

    // Read should have succeeded (n >= 0) or hit EOF (n == 0)
    return n >= 0;
}

// Test dup shared file position - POSIX semantics require dup'd fds to share position
// Returns true if position is shared correctly, false otherwise.
// Content should be "ABCDEFGH" (8 bytes)
bool test_dup_shared_position(const char *path) {
    if (!hl_fs_initialized()) {
        return false;
    }

    // Open file for reading
    int fd1 = open(path, hl_HL_O_RDONLY);
    if (fd1 < 0) {
        return false;
    }

    // Duplicate fd
    int fd2 = dup(fd1);
    if (fd2 < 0) {
        close(fd1);
        return false;
    }

    // Read 2 bytes from fd1: should get "AB", position now at 2
    char buf1[3] = {0};
    if (read(fd1, buf1, 2) != 2) {
        close(fd1);
        close(fd2);
        return false;
    }

    // Read 2 bytes from fd2: position is shared, should get "CD" (pos 2-4)
    char buf2[3] = {0};
    if (read(fd2, buf2, 2) != 2) {
        close(fd1);
        close(fd2);
        return false;
    }

    close(fd1);
    close(fd2);

    // Verify: fd1 read "AB", fd2 read "CD" (shared position)
    return (buf1[0] == 'A' && buf1[1] == 'B' &&
            buf2[0] == 'C' && buf2[1] == 'D');
}

// Test mkdirat with AT_FDCWD
int test_mkdirat_cwd(const char *path) {
    if (!hl_fs_initialized()) {
        return -1;
    }

    // Create directory using mkdirat with AT_FDCWD
    int result = mkdirat(hl_HL_AT_FDCWD, path, 0755);
    if (result < 0) {
        return result;  // Error code
    }

    // Verify it exists
    hl_Stat st;
    if (stat(path, &st) < 0 || !st.is_dir) {
        // Should never happen: mkdir succeeded but stat fails or not a dir
        return -2;
    }

    // Clean up
    rmdir(path);
    return 1;  // Success
}

HYPERLIGHT_WRAP_FUNCTION(test_opendir_readdir, Int, 1, String)
HYPERLIGHT_WRAP_FUNCTION(test_opendir_list, String, 1, String)
HYPERLIGHT_WRAP_FUNCTION(test_access, Int, 2, String, Int)
HYPERLIGHT_WRAP_FUNCTION(test_openat_cwd, Int, 2, String, Int)
HYPERLIGHT_WRAP_FUNCTION(test_fcntl_flags, Int, 1, String)
HYPERLIGHT_WRAP_FUNCTION(test_fd_lifetime_across_function_return, Int, 1, String)
HYPERLIGHT_WRAP_FUNCTION(test_dup, Bool, 1, String)
HYPERLIGHT_WRAP_FUNCTION(test_dup2, Bool, 1, String)
HYPERLIGHT_WRAP_FUNCTION(test_dup_shared_position, Bool, 1, String)
HYPERLIGHT_WRAP_FUNCTION(test_mkdirat_cwd, Int, 1, String)

// =============================================================================
// Open Flag Tests - Testing O_TRUNC, O_EXCL, O_APPEND, O_CREAT
// =============================================================================

// Test O_TRUNC: Opening an existing file with O_TRUNC should truncate it.
// Writes "ORIGINAL", then opens with O_TRUNC and writes "NEW", verifies only "NEW" remains.
// Returns 1 on success, negative error code on failure.
int test_o_trunc(const char *path) {
    if (!hl_fs_initialized()) {
        return -1;
    }

    // First, create a file with some content
    int fd = open(path, hl_HL_O_WRONLY | hl_HL_O_CREAT | hl_HL_O_TRUNC);
    if (fd < 0) {
        return -2;
    }
    const char *original = "ORIGINAL_LONG_CONTENT";
    write(fd, original, strlen(original));
    close(fd);

    // Now open with O_TRUNC and write shorter content
    fd = open(path, hl_HL_O_WRONLY | hl_HL_O_TRUNC);
    if (fd < 0) {
        return -3;
    }
    const char *new_content = "NEW";
    write(fd, new_content, strlen(new_content));
    close(fd);

    // Read back and verify only "NEW" is there (not "NEWGINAL_LONG_CONTENT")
    fd = open(path, hl_HL_O_RDONLY);
    if (fd < 0) {
        return -4;
    }
    char buf[64] = {0};
    int64_t bytes_read = read(fd, buf, sizeof(buf) - 1);
    close(fd);

    if (bytes_read != 3) {
        return -5;  // Wrong length - O_TRUNC didn't work
    }
    if (strcmp(buf, "NEW") != 0) {
        return -6;  // Wrong content
    }

    // Clean up
    unlink(path);
    return 1;  // Success
}

// Test O_EXCL: Opening with O_CREAT | O_EXCL should fail if file exists.
// Returns 1 on success (correct behavior), negative error code on failure.
int test_o_excl(const char *path) {
    if (!hl_fs_initialized()) {
        return -1;
    }

    // First, create a file
    int fd = open(path, hl_HL_O_WRONLY | hl_HL_O_CREAT | hl_HL_O_TRUNC);
    if (fd < 0) {
        return -2;
    }
    write(fd, "EXISTS", 6);
    close(fd);

    // Now try to open with O_CREAT | O_EXCL - should FAIL because file exists
    fd = open(path, hl_HL_O_WRONLY | hl_HL_O_CREAT | hl_HL_O_EXCL);
    if (fd >= 0) {
        // Should have failed!
        close(fd);
        unlink(path);
        return -3;  // O_EXCL didn't reject existing file
    }

    // Good - it failed as expected. Now verify the original content is untouched.
    fd = open(path, hl_HL_O_RDONLY);
    if (fd < 0) {
        return -4;
    }
    char buf[32] = {0};
    read(fd, buf, sizeof(buf) - 1);
    close(fd);

    if (strcmp(buf, "EXISTS") != 0) {
        return -5;  // File was somehow modified
    }

    // Clean up
    unlink(path);
    return 1;  // Success - O_EXCL correctly rejected existing file
}

// Test O_EXCL with new file: O_CREAT | O_EXCL should succeed for new file.
// Returns 1 on success, negative error code on failure.
int test_o_excl_new_file(const char *path) {
    if (!hl_fs_initialized()) {
        return -1;
    }

    // Make sure file doesn't exist first
    unlink(path);

    // Open with O_CREAT | O_EXCL - should succeed for new file
    int fd = open(path, hl_HL_O_WRONLY | hl_HL_O_CREAT | hl_HL_O_EXCL);
    if (fd < 0) {
        return -2;  // Should have succeeded for new file
    }

    write(fd, "CREATED", 7);
    close(fd);

    // Verify content
    fd = open(path, hl_HL_O_RDONLY);
    if (fd < 0) {
        return -3;
    }
    char buf[32] = {0};
    read(fd, buf, sizeof(buf) - 1);
    close(fd);

    if (strcmp(buf, "CREATED") != 0) {
        unlink(path);
        return -4;
    }

    // Clean up
    unlink(path);
    return 1;  // Success
}

// Test O_EXCL without O_CREAT: Should return EINVAL (undefined per POSIX, we reject it).
// Returns 1 on success (EINVAL returned), negative error code on failure.
int test_o_excl_no_creat(const char *path) {
    if (!hl_fs_initialized()) {
        return -1;
    }

    // First create a file so we have something to try opening
    int fd = open(path, hl_HL_O_WRONLY | hl_HL_O_CREAT | hl_HL_O_TRUNC);
    if (fd < 0) {
        return -2;
    }
    write(fd, "TEST", 4);
    close(fd);

    // Now try to open with O_EXCL but WITHOUT O_CREAT - should fail with EINVAL
    fd = open(path, hl_HL_O_WRONLY | hl_HL_O_EXCL);
    if (fd >= 0) {
        // Should have failed!
        close(fd);
        unlink(path);
        return -3;  // O_EXCL without O_CREAT should have been rejected
    }

    // Verify we got EINVAL (or at least some error)
    // The fd should be negative error code
    if (fd != hl_HL_EINVAL) {
        unlink(path);
        return -4;  // Wrong error code
    }

    // Clean up
    unlink(path);
    return 1;  // Success - O_EXCL without O_CREAT correctly returned EINVAL
}

// Test O_APPEND: Writes should always go to end of file.
// Returns 1 on success, negative error code on failure.
int test_o_append(const char *path) {
    if (!hl_fs_initialized()) {
        return -1;
    }

    // Create a file with initial content
    int fd = open(path, hl_HL_O_WRONLY | hl_HL_O_CREAT | hl_HL_O_TRUNC);
    if (fd < 0) {
        return -2;
    }
    write(fd, "FIRST", 5);
    close(fd);

    // Open with O_APPEND and write more
    fd = open(path, hl_HL_O_WRONLY | hl_HL_O_APPEND);
    if (fd < 0) {
        return -3;
    }
    write(fd, "SECOND", 6);
    close(fd);

    // Read back and verify both parts are there
    fd = open(path, hl_HL_O_RDONLY);
    if (fd < 0) {
        return -4;
    }
    char buf[64] = {0};
    int64_t bytes_read = read(fd, buf, sizeof(buf) - 1);
    close(fd);

    if (bytes_read != 11) {
        unlink(path);
        return -5;  // Wrong length
    }
    if (strcmp(buf, "FIRSTSECOND") != 0) {
        unlink(path);
        return -6;  // Content wrong - append didn't work
    }

    // Clean up
    unlink(path);
    return 1;  // Success
}

// Test O_CREAT: Create file if it doesn't exist.
// Returns 1 on success, negative error code on failure.
int test_o_creat(const char *path) {
    if (!hl_fs_initialized()) {
        return -1;
    }

    // Make sure file doesn't exist
    unlink(path);

    // Verify file doesn't exist
    hl_Stat st;
    if (stat(path, &st) >= 0) {
        return -2;  // File shouldn't exist yet
    }

    // Open with O_CREAT - should create it
    int fd = open(path, hl_HL_O_WRONLY | hl_HL_O_CREAT);
    if (fd < 0) {
        return -3;
    }
    write(fd, "CREATED_BY_O_CREAT", 18);
    close(fd);

    // Verify file now exists and has content
    if (stat(path, &st) < 0) {
        return -4;  // File should exist now
    }

    fd = open(path, hl_HL_O_RDONLY);
    if (fd < 0) {
        return -5;
    }
    char buf[64] = {0};
    read(fd, buf, sizeof(buf) - 1);
    close(fd);

    if (strcmp(buf, "CREATED_BY_O_CREAT") != 0) {
        unlink(path);
        return -6;
    }

    // Clean up
    unlink(path);
    return 1;  // Success
}

// Test O_RDWR: Can both read and write to file.
// Returns 1 on success, negative error code on failure.
int test_o_rdwr(const char *path) {
    if (!hl_fs_initialized()) {
        return -1;
    }

    // Create file with O_RDWR | O_CREAT | O_TRUNC
    int fd = open(path, hl_HL_O_RDWR | hl_HL_O_CREAT | hl_HL_O_TRUNC);
    if (fd < 0) {
        return -2;
    }

    // Write some data
    write(fd, "RDWR_TEST", 9);

    // Seek back to beginning
    lseek(fd, 0, hl_HL_SEEK_SET);

    // Read it back using the same fd
    char buf[32] = {0};
    int64_t bytes_read = read(fd, buf, sizeof(buf) - 1);
    close(fd);

    if (bytes_read != 9) {
        unlink(path);
        return -3;
    }
    if (strcmp(buf, "RDWR_TEST") != 0) {
        unlink(path);
        return -4;
    }

    // Clean up
    unlink(path);
    return 1;  // Success
}

// Test O_APPEND after lseek: Writes should still go to end even after lseek.
// This verifies POSIX O_APPEND semantics where each write seeks to EOF first.
// Returns 1 on success, negative error code on failure.
int test_o_append_after_lseek(const char *path) {
    if (!hl_fs_initialized()) {
        return -1;
    }

    // Create file with initial content
    int fd = open(path, hl_HL_O_WRONLY | hl_HL_O_CREAT | hl_HL_O_TRUNC);
    if (fd < 0) {
        return -2;
    }
    write(fd, "FIRST", 5);
    close(fd);

    // Open with O_APPEND
    fd = open(path, hl_HL_O_WRONLY | hl_HL_O_APPEND);
    if (fd < 0) {
        return -3;
    }

    // Seek to beginning - this should be ignored for writes due to O_APPEND
    lseek(fd, 0, hl_HL_SEEK_SET);

    // Write more data - should go to END, not position 0
    write(fd, "SECOND", 6);
    close(fd);

    // Read back and verify content
    fd = open(path, hl_HL_O_RDONLY);
    if (fd < 0) {
        return -4;
    }
    char buf[64] = {0};
    int64_t bytes_read = read(fd, (uint8_t*)buf, sizeof(buf) - 1);
    close(fd);

    if (bytes_read != 11) {
        unlink(path);
        return -5;  // Wrong length - O_APPEND might have been ignored
    }
    if (strcmp(buf, "FIRSTSECOND") != 0) {
        unlink(path);
        return -6;  // Content wrong - write went to position 0 instead of end
    }

    unlink(path);
    return 1;  // Success - O_APPEND works correctly after lseek
}

// Test fcntl F_GETFL returns accurate flags including O_APPEND.
// Returns 1 on success, negative error code on failure.
int test_fcntl_getfl_accuracy(const char *path) {
    if (!hl_fs_initialized()) {
        return -1;
    }

    // Create file
    int fd = open(path, hl_HL_O_RDWR | hl_HL_O_CREAT | hl_HL_O_TRUNC);
    if (fd < 0) {
        return -2;
    }
    close(fd);

    // Open with O_WRONLY | O_APPEND
    fd = open(path, hl_HL_O_WRONLY | hl_HL_O_APPEND);
    if (fd < 0) {
        return -3;
    }

    // Get flags with fcntl
    int flags = hl_fs_fcntl(fd, hl_HL_F_GETFL, 0);
    close(fd);

    // Verify O_WRONLY is set
    if ((flags & 0x3) != hl_HL_O_WRONLY) {
        unlink(path);
        return -4;  // Access mode wrong
    }

    // Verify O_APPEND is set
    if ((flags & hl_HL_O_APPEND) == 0) {
        unlink(path);
        return -5;  // O_APPEND not returned by F_GETFL
    }

    unlink(path);
    return 1;  // Success
}

// Test fcntl F_SETFL can enable O_APPEND on a file not opened with it.
// Returns 1 on success, negative error code on failure.
int test_fcntl_setfl_append(const char *path) {
    if (!hl_fs_initialized()) {
        return -1;
    }

    // Create file with initial content
    int fd = open(path, hl_HL_O_WRONLY | hl_HL_O_CREAT | hl_HL_O_TRUNC);
    if (fd < 0) {
        return -2;
    }
    write(fd, "FIRST", 5);
    close(fd);

    // Open WITHOUT O_APPEND
    fd = open(path, hl_HL_O_WRONLY);
    if (fd < 0) {
        return -3;
    }

    // Use F_SETFL to enable O_APPEND
    hl_fs_fcntl(fd, hl_HL_F_SETFL, hl_HL_O_APPEND);

    // Verify F_GETFL now shows O_APPEND
    int flags = hl_fs_fcntl(fd, hl_HL_F_GETFL, 0);
    if ((flags & hl_HL_O_APPEND) == 0) {
        close(fd);
        unlink(path);
        return -4;  // F_SETFL didn't enable O_APPEND
    }

    // Write - should go to end due to dynamically enabled O_APPEND
    write(fd, "SECOND", 6);
    close(fd);

    // Read back and verify
    fd = open(path, hl_HL_O_RDONLY);
    if (fd < 0) {
        return -5;
    }
    char buf[64] = {0};
    read(fd, (uint8_t*)buf, sizeof(buf) - 1);
    close(fd);

    if (strcmp(buf, "FIRSTSECOND") != 0) {
        unlink(path);
        return -6;  // F_SETFL O_APPEND didn't work
    }

    unlink(path);
    return 1;  // Success
}

// Test openat with real directory fd returns ENOTSUP.
// Returns 1 on success (ENOTSUP returned), negative error code on failure.
int test_openat_real_dirfd(const char *path) {
    if (!hl_fs_initialized()) {
        return -1;
    }

    // First create a file to open
    int fd = open(path, hl_HL_O_WRONLY | hl_HL_O_CREAT | hl_HL_O_TRUNC);
    if (fd < 0) {
        return -2;
    }
    close(fd);

    // Try to use the fd (which is a file, not directory) as dirfd
    // This should return ENOTSUP since we only support AT_FDCWD
    int result = hl_fs_openat(fd, "somefile.txt", hl_HL_O_RDONLY);

    unlink(path);

    // We expect ENOTSUP (-2) because non-AT_FDCWD dirfd is not supported
    if (result == -2) {  // HL_ENOTSUP
        return 1;  // Success - correctly returned ENOTSUP
    }
    return -3;  // Wrong error or unexpected success
}

HYPERLIGHT_WRAP_FUNCTION(test_o_trunc, Int, 1, String)
HYPERLIGHT_WRAP_FUNCTION(test_o_excl, Int, 1, String)
HYPERLIGHT_WRAP_FUNCTION(test_o_excl_new_file, Int, 1, String)
HYPERLIGHT_WRAP_FUNCTION(test_o_excl_no_creat, Int, 1, String)
HYPERLIGHT_WRAP_FUNCTION(test_o_append, Int, 1, String)
HYPERLIGHT_WRAP_FUNCTION(test_o_creat, Int, 1, String)
HYPERLIGHT_WRAP_FUNCTION(test_o_rdwr, Int, 1, String)
HYPERLIGHT_WRAP_FUNCTION(test_o_append_after_lseek, Int, 1, String)
HYPERLIGHT_WRAP_FUNCTION(test_fcntl_getfl_accuracy, Int, 1, String)
HYPERLIGHT_WRAP_FUNCTION(test_fcntl_setfl_append, Int, 1, String)
HYPERLIGHT_WRAP_FUNCTION(test_openat_real_dirfd, Int, 1, String)

HYPERLIGHT_WRAP_FUNCTION(delete_fat_file, Bool, 1, String)
HYPERLIGHT_WRAP_FUNCTION(mkdir_fat, Bool, 1, String)
HYPERLIGHT_WRAP_FUNCTION(rmdir_fat, Bool, 1, String)
HYPERLIGHT_WRAP_FUNCTION(list_dir_fat, String, 1, String)
HYPERLIGHT_WRAP_FUNCTION(rename_fat, Bool, 2, String, String)
HYPERLIGHT_WRAP_FUNCTION(get_cwd, String, 0)
HYPERLIGHT_WRAP_FUNCTION(do_chdir, Bool, 1, String)
HYPERLIGHT_WRAP_FUNCTION(stat_fat_size, Long, 1, String)
HYPERLIGHT_WRAP_FUNCTION(exists_fat, Int, 1, String)

// Guest-created FAT mount functions
HYPERLIGHT_WRAP_FUNCTION(create_fat_mount, Bool, 2, String, Long)
HYPERLIGHT_WRAP_FUNCTION(unmount_fat, Bool, 1, String)
HYPERLIGHT_WRAP_FUNCTION(is_guest_created_mount, Bool, 1, String)
HYPERLIGHT_WRAP_FUNCTION(test_guest_fat_full_cycle, Int, 2, String, Long)
HYPERLIGHT_WRAP_FUNCTION(test_unmount_with_open_file, Int, 2, String, Long)

HYPERLIGHT_WRAP_FUNCTION(is_fs_initialized, Int, 0)
HYPERLIGHT_WRAP_FUNCTION(guest_fn_checks_if_host_returns_float_value, Float, 2, Float, Float)
HYPERLIGHT_WRAP_FUNCTION(guest_fn_checks_if_host_returns_double_value, Double, 2, Double, Double)
HYPERLIGHT_WRAP_FUNCTION(guest_fn_checks_if_host_returns_string_value, String, 0)
HYPERLIGHT_WRAP_FUNCTION(guest_fn_checks_if_host_returns_bool_value, Bool, 2, Int, Int)
HYPERLIGHT_WRAP_FUNCTION(echo, String, 1, String)
// HYPERLIGHT_WRAP_FUNCTION(set_byte_array_to_zero, 1, VecBytes) is not valid for functions that return VecBytes
HYPERLIGHT_WRAP_FUNCTION(guest_function, Int, 1, String)
HYPERLIGHT_WRAP_FUNCTION(print_output, Int, 1, String)
HYPERLIGHT_WRAP_FUNCTION(stack_allocate, Int, 1, Int)
HYPERLIGHT_WRAP_FUNCTION(stack_overflow, Int, 1, Int)
HYPERLIGHT_WRAP_FUNCTION(buffer_overrun, Int, 1, String)
HYPERLIGHT_WRAP_FUNCTION(large_var, Int, 0)
HYPERLIGHT_WRAP_FUNCTION(small_var, Int, 0) 
HYPERLIGHT_WRAP_FUNCTION(call_malloc, Int, 1, Int)
HYPERLIGHT_WRAP_FUNCTION(malloc_and_free, Int, 1, Int)
HYPERLIGHT_WRAP_FUNCTION(print_two_args, Int, 2, String, Int)
HYPERLIGHT_WRAP_FUNCTION(print_three_args, Int, 3, String, Int, Long)
HYPERLIGHT_WRAP_FUNCTION(print_four_args, Int, 4, String, Int, Long, String)
HYPERLIGHT_WRAP_FUNCTION(print_five_args, Int, 5, String, Int, Long, String, String)
HYPERLIGHT_WRAP_FUNCTION(print_six_args, Int, 6, String, Int, Long, String, String, Bool)
HYPERLIGHT_WRAP_FUNCTION(print_seven_args, Int, 7, String, Int, Long, String, String, Bool, Bool)
HYPERLIGHT_WRAP_FUNCTION(print_eight_args, Int, 8, String, Int, Long, String, String, Bool, Bool, UInt)
HYPERLIGHT_WRAP_FUNCTION(print_nine_args, Int, 9, String, Int, Long, String, String, Bool, Bool, UInt, ULong)
HYPERLIGHT_WRAP_FUNCTION(print_ten_args, Int, 10, String, Int, Long, String, String, Bool, Bool, UInt, ULong, Int)
HYPERLIGHT_WRAP_FUNCTION(print_eleven_args, Int, 11, String, Int, Long, String, String, Bool, Bool, UInt, ULong, Int, Float)
HYPERLIGHT_WRAP_FUNCTION(echo_float, Float, 1, Float)
HYPERLIGHT_WRAP_FUNCTION(echo_double, Double, 1, Double)
HYPERLIGHT_WRAP_FUNCTION(set_static, Int, 0)
// HYPERLIGHT_WRAP_FUNCTION(get_size_prefixed_buffer, Int, 1, VecBytes) is not valid for functions that return VecBytes
HYPERLIGHT_WRAP_FUNCTION(guest_abort_with_msg, Int, 2, Int, String)
HYPERLIGHT_WRAP_FUNCTION(guest_abort_with_code, Int, 1, Int)
HYPERLIGHT_WRAP_FUNCTION(execute_on_stack, Int, 0)
HYPERLIGHT_WRAP_FUNCTION(log_message, Int, 2, String, Long)
// HYPERLIGHT_WRAP_FUNCTION(twenty_four_k_in_eight_k_out, VecBytes, 1, VecBytes) is not valid for functions that return VecBytes

void hyperlight_main(void)
{
    HYPERLIGHT_REGISTER_FUNCTION("GuestRetrievesFloatValue", guest_fn_checks_if_host_returns_float_value);
    HYPERLIGHT_REGISTER_FUNCTION("GuestRetrievesDoubleValue", guest_fn_checks_if_host_returns_double_value);
    HYPERLIGHT_REGISTER_FUNCTION("GuestRetrievesStringValue", guest_fn_checks_if_host_returns_string_value);
    HYPERLIGHT_REGISTER_FUNCTION("GuestRetrievesBoolValue", guest_fn_checks_if_host_returns_bool_value);
    HYPERLIGHT_REGISTER_FUNCTION("Echo", echo);
    // HYPERLIGHT_REGISTER_FUNCTION macro does not work for functions that return VecBytes,
    // so we use hl_register_function_definition directly
    hl_register_function_definition("SetByteArrayToZero", set_byte_array_to_zero, 1, (hl_ParameterType[]){hl_ParameterType_VecBytes}, hl_ReturnType_VecBytes);
    HYPERLIGHT_REGISTER_FUNCTION("GuestMethod1", guest_function);
    HYPERLIGHT_REGISTER_FUNCTION("PrintOutput", print_output);
    HYPERLIGHT_REGISTER_FUNCTION("StackAllocate", stack_allocate);
    HYPERLIGHT_REGISTER_FUNCTION("StackOverflow", stack_overflow);
    HYPERLIGHT_REGISTER_FUNCTION("BufferOverrun", buffer_overrun);
    HYPERLIGHT_REGISTER_FUNCTION("LargeVar", large_var);
    HYPERLIGHT_REGISTER_FUNCTION("SmallVar", small_var);
    HYPERLIGHT_REGISTER_FUNCTION("CallMalloc", call_malloc);
    HYPERLIGHT_REGISTER_FUNCTION("MallocAndFree", malloc_and_free);
    HYPERLIGHT_REGISTER_FUNCTION("PrintTwoArgs", print_two_args);
    HYPERLIGHT_REGISTER_FUNCTION("PrintThreeArgs", print_three_args);
    HYPERLIGHT_REGISTER_FUNCTION("PrintFourArgs", print_four_args);
    HYPERLIGHT_REGISTER_FUNCTION("PrintFiveArgs", print_five_args);
    HYPERLIGHT_REGISTER_FUNCTION("PrintSixArgs", print_six_args);
    HYPERLIGHT_REGISTER_FUNCTION("PrintSevenArgs", print_seven_args);
    HYPERLIGHT_REGISTER_FUNCTION("PrintEightArgs", print_eight_args);
    HYPERLIGHT_REGISTER_FUNCTION("PrintNineArgs", print_nine_args);
    HYPERLIGHT_REGISTER_FUNCTION("PrintTenArgs", print_ten_args);
    HYPERLIGHT_REGISTER_FUNCTION("PrintElevenArgs", print_eleven_args);
    HYPERLIGHT_REGISTER_FUNCTION("EchoFloat", echo_float);
    HYPERLIGHT_REGISTER_FUNCTION("EchoDouble", echo_double);
    HYPERLIGHT_REGISTER_FUNCTION("SetStatic", set_static);
    // HYPERLIGHT_REGISTER_FUNCTION macro does not work for functions that return VecBytes,
    // so we use hl_register_function_definition directly
    hl_register_function_definition("GetSizePrefixedBuffer", get_size_prefixed_buffer, 1, (hl_ParameterType[]){hl_ParameterType_VecBytes}, hl_ReturnType_VecBytes);
    HYPERLIGHT_REGISTER_FUNCTION("GuestAbortWithCode", guest_abort_with_code);
    HYPERLIGHT_REGISTER_FUNCTION("GuestAbortWithMessage", guest_abort_with_msg);
    HYPERLIGHT_REGISTER_FUNCTION("ExecuteOnStack", execute_on_stack);
    HYPERLIGHT_REGISTER_FUNCTION("LogMessage", log_message);
    // HYPERLIGHT_REGISTER_FUNCTION macro does not work for functions that return VecBytes,
    // so we use hl_register_function_definition directly
    hl_register_function_definition("24K_in_8K_out", twenty_four_k_in_eight_k_out, 1, (hl_ParameterType[]){hl_ParameterType_VecBytes}, hl_ReturnType_VecBytes);
    // HyperlightFS functions
    HYPERLIGHT_REGISTER_FUNCTION("IsFsInitialized", is_fs_initialized);
    hl_register_function_definition("ReadFile", read_file, 1, (hl_ParameterType[]){hl_ParameterType_String}, hl_ReturnType_VecBytes);
    hl_register_function_definition("RandomReadChunks", random_read_chunks, 1, (hl_ParameterType[]){hl_ParameterType_String}, hl_ReturnType_VecBytes);
    // FAT filesystem functions
    hl_register_function_definition("WriteFatFile", write_fat_file, 2, (hl_ParameterType[]){hl_ParameterType_String, hl_ParameterType_VecBytes}, hl_ReturnType_Bool);
    hl_register_function_definition("ReadFatFile", read_fat_file, 1, (hl_ParameterType[]){hl_ParameterType_String}, hl_ReturnType_VecBytes);
    HYPERLIGHT_REGISTER_FUNCTION("DeleteFatFile", delete_fat_file);
    HYPERLIGHT_REGISTER_FUNCTION("MkdirFat", mkdir_fat);
    HYPERLIGHT_REGISTER_FUNCTION("RmdirFat", rmdir_fat);
    HYPERLIGHT_REGISTER_FUNCTION("ListDirFat", list_dir_fat);
    HYPERLIGHT_REGISTER_FUNCTION("RenameFat", rename_fat);
    HYPERLIGHT_REGISTER_FUNCTION("StatFatSize", stat_fat_size);
    HYPERLIGHT_REGISTER_FUNCTION("ExistsFat", exists_fat);
    // Guest-created FAT mount functions
    HYPERLIGHT_REGISTER_FUNCTION("CreateFatMount", create_fat_mount);
    HYPERLIGHT_REGISTER_FUNCTION("UnmountFat", unmount_fat);
    HYPERLIGHT_REGISTER_FUNCTION("IsGuestCreatedMount", is_guest_created_mount);
    HYPERLIGHT_REGISTER_FUNCTION("TestGuestFatFullCycle", test_guest_fat_full_cycle);
    HYPERLIGHT_REGISTER_FUNCTION("TestUnmountWithOpenFile", test_unmount_with_open_file);
    // CWD functions
    HYPERLIGHT_REGISTER_FUNCTION("GetCwd", get_cwd);
    HYPERLIGHT_REGISTER_FUNCTION("Chdir", do_chdir);
    // New C API test functions
    HYPERLIGHT_REGISTER_FUNCTION("TestOpendirReaddir", test_opendir_readdir);
    HYPERLIGHT_REGISTER_FUNCTION("TestOpendirList", test_opendir_list);
    HYPERLIGHT_REGISTER_FUNCTION("TestAccess", test_access);
    HYPERLIGHT_REGISTER_FUNCTION("TestOpenatCwd", test_openat_cwd);
    HYPERLIGHT_REGISTER_FUNCTION("TestFcntlFlags", test_fcntl_flags);
    HYPERLIGHT_REGISTER_FUNCTION("TestFdLifetimeAcrossFunctionReturn", test_fd_lifetime_across_function_return);
    HYPERLIGHT_REGISTER_FUNCTION("TestDup", test_dup);
    HYPERLIGHT_REGISTER_FUNCTION("TestDup2", test_dup2);
    HYPERLIGHT_REGISTER_FUNCTION("TestDupSharedPosition", test_dup_shared_position);
    HYPERLIGHT_REGISTER_FUNCTION("TestMkdiratCwd", test_mkdirat_cwd);
    // Open flag tests
    HYPERLIGHT_REGISTER_FUNCTION("TestOTrunc", test_o_trunc);
    HYPERLIGHT_REGISTER_FUNCTION("TestOExcl", test_o_excl);
    HYPERLIGHT_REGISTER_FUNCTION("TestOExclNewFile", test_o_excl_new_file);
    HYPERLIGHT_REGISTER_FUNCTION("TestOExclNoCreat", test_o_excl_no_creat);
    HYPERLIGHT_REGISTER_FUNCTION("TestOAppend", test_o_append);
    HYPERLIGHT_REGISTER_FUNCTION("TestOCreat", test_o_creat);
    HYPERLIGHT_REGISTER_FUNCTION("TestORdwr", test_o_rdwr);
    // POSIX compliance tests
    HYPERLIGHT_REGISTER_FUNCTION("TestOAppendAfterLseek", test_o_append_after_lseek);
    HYPERLIGHT_REGISTER_FUNCTION("TestFcntlGetflAccuracy", test_fcntl_getfl_accuracy);
    HYPERLIGHT_REGISTER_FUNCTION("TestFcntlSetflAppend", test_fcntl_setfl_append);
    HYPERLIGHT_REGISTER_FUNCTION("TestOpenatRealDirfd", test_openat_real_dirfd);
}

// This dispatch function is only used when the host dispatches a guest function
// call but there is no registered guest function with the given name.
hl_Vec *c_guest_dispatch_function(const hl_FunctionCall *function_call) {
  const char *func_name = function_call->function_name;
  if (strcmp(func_name, "ThisIsNotARealFunctionButTheNameIsImportant") == 0) {
    // TODO DO A LOG HERE
    // This is special case for test `iostack_is_working
    return hl_flatbuffer_result_from_Int(99);
  }

  return NULL;
}
