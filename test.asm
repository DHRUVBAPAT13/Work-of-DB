%macro rw 4
    mov rax,%1
    mov rdi,%2
    mov rsi,%3
    mov rdx,%4
    syscall
%endmacro

section .data 
    msg db 'hello this is dhruv',10
    len1 equ $-msg

section .text

global _start

_start:

    rw 1,1,msg,len1
    rw 60,0,0,0