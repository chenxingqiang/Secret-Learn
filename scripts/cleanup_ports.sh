#!/bin/bash
# 清理 SecretFlow 占用的端口

echo "=========================================="
echo "SecretFlow 端口清理工具"
echo "=========================================="
echo ""

# 定义端口范围
PORTS="9491 9492 9493 9494 9495 9496 9497 9498 9499"

echo "检查占用的端口..."
echo ""

FOUND=0
for PORT in $PORTS; do
    PID=$(lsof -ti :$PORT 2>/dev/null)
    if [ ! -z "$PID" ]; then
        echo "  端口 $PORT 被进程 $PID 占用"
        FOUND=1
    fi
done

if [ $FOUND -eq 0 ]; then
    echo "  所有端口空闲"
    echo ""
    exit 0
fi

echo ""
echo "是否清理这些进程？(y/n)"
read -r response

if [ "$response" = "y" ] || [ "$response" = "yes" ]; then
    echo ""
    echo "清理中..."
    for PORT in $PORTS; do
        PID=$(lsof -ti :$PORT 2>/dev/null)
        if [ ! -z "$PID" ]; then
            kill -9 $PID 2>/dev/null
            echo "  清理端口 $PORT (PID: $PID)"
        fi
    done
    echo ""
    echo "清理完成！"
else
    echo ""
    echo "操作已取消"
fi

echo ""
echo "=========================================="

