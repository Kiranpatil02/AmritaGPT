import React from "react";
import UserChat from "./UserChat";
import GPTChat from "./GPTChat";

const MessageContainer = ({ messages  =[]}) => (
  console.log(messages),
  <div>
    <div className="flex flex-col ">
      <GPTChat message={"Hey im amrita gpt"} />
      <UserChat message={"Tell me about the clubs in Amrita!"} />
      <GPTChat message={"Based on the provided context, here are all the clubs in Amrita that I can find information about:  IETE Club at Amrita, Google Developer Student Clubs Amrita, Srishti - The literary club, Natyasudha (Dance Club), Ragasudha (Music Club)"} />
    </div>
    <div className="flex flex-col space-y-4 p-4">
      {messages.map((message, index) => (
        message.user ? (
          <UserChat key={index} message={message.text} />
        ) : (
          <GPTChat key={index} message={message.text} />
        )
      ))}
    </div>
  </div>
);
export default MessageContainer;
